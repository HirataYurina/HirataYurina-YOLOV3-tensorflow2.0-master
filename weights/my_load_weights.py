# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:my_load_weights.py
# software: PyCharm

import warnings
import tensorflow.keras as keras
import h5py
import numpy as np


def convert_kernel(kernel):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

    Also works reciprocally, since the transformation is its own inverse.

    # Arguments
        kernel: Numpy array (3D, 4D or 5D).

    # Returns
        The converted kernel.

    # Raises
        ValueError: in case of invalid kernel shape or invalid data_format.
    """
    kernel = np.asarray(kernel)
    if not 3 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[-2:] = no_flip
    return np.copy(kernel[tuple(slices)])


def _need_convert_kernel(original_backend):
    """Checks if conversion on kernel matrices is required during weight loading.

    The convolution operation is implemented differently in different backends.
    While TH implements convolution, TF and CNTK implement the correlation operation.
    So the channel axis needs to be flipped when TF weights are loaded on a TH model,
    or vice versa. However, there's no conversion required between TF and CNTK.

    # Arguments
        original_backend: Keras backend the weights were trained with, as a string.

    # Returns
        `True` if conversion on kernel matrices is required, otherwise `False`.
    """
    if original_backend is None:
        # backend information not available
        return False
    uses_correlation = {'tensorflow': True,
                        'theano': False,
                        'cntk': True}
    if original_backend not in uses_correlation:
        # By default, do not convert the kernels if the original backend is unknown
        return False
    if keras.backend.backend() in uses_correlation:
        current_uses_correlation = uses_correlation[keras.backend.backend()]
    else:
        # Assume unknown backends use correlation
        current_uses_correlation = True
    return uses_correlation[original_backend] != current_uses_correlation


def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    # Arguments
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    # Returns
        data: Attributes data.
    """
    if name in group.attrs:
        data = [n.decode('utf8') for n in group.attrs[name]]
    else:
        data = []
        chunk_id = 0
        while ('%s%d' % (name, chunk_id)) in group.attrs:
            data.extend([n.decode('utf8')
                         for n in group.attrs['%s%d' % (name, chunk_id)]])
            chunk_id += 1
    return data


def load_weights_from_hdf5_group(f, layers, reshape=False):
    """Implements topological (order-based) weight loading.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' +
                         str(len(filtered_layers)) + ' layers.')

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = preprocess_weights_for_loading(layer,
                                                       weight_values,
                                                       original_keras_version,
                                                       original_backend,
                                                       reshape=reshape)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) +
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    keras.backend.batch_set_value(weight_value_tuples)


def _convert_rnn_weights(layer, weights):
    """Converts weights for RNN layers between native and CuDNN format.

    Input kernels for each gate are transposed and converted between Fortran
    and C layout, recurrent kernels are transposed. For LSTM biases are summed/
    split in half, for GRU biases are reshaped.

    Weights can be converted in both directions between `LSTM` and`CuDNNSLTM`
    and between `CuDNNGRU` and `GRU(reset_after=True)`. Default `GRU` is not
    compatible with `CuDNNGRU`.

    For missing biases in `LSTM`/`GRU` (`use_bias=False`),
    no conversion is made.

    # Arguments
        layer: Target layer instance.
        weights: List of source weights values (input kernels, recurrent
            kernels, [biases]) (Numpy arrays).

    # Returns
        A list of converted weights values (Numpy arrays).

    # Raises
        ValueError: for incompatible GRU layer/weights or incompatible biases
    """

    def transform_kernels(kernels, func, n_gates):
        """Transforms kernel for each gate separately using given function.

        # Arguments
            kernels: Stacked array of kernels for individual gates.
            func: Function applied to kernel of each gate.
            n_gates: Number of gates (4 for LSTM, 3 for GRU).
        # Returns
            Stacked array of transformed kernels.
        """
        return np.hstack([func(k) for k in np.hsplit(kernels, n_gates)])

    def transpose_input(from_cudnn):
        """Makes a function that transforms input kernels from/to CuDNN format.

        It keeps the shape, but changes between the layout (Fortran/C). Eg.:

        ```
        Keras                 CuDNN
        [[0, 1, 2],  <--->  [[0, 2, 4],
         [3, 4, 5]]          [1, 3, 5]]
        ```

        It can be passed to `transform_kernels()`.

        # Arguments
            from_cudnn: `True` if source weights are in CuDNN format, `False`
                if they're in plain Keras format.
        # Returns
            Function that converts input kernel to the other format.
        """
        order = 'F' if from_cudnn else 'C'

        def transform(kernel):
            return kernel.T.reshape(kernel.shape, order=order)

        return transform

    target_class = layer.__class__.__name__

    # convert the weights between CuDNNLSTM and LSTM
    if target_class in ['LSTM', 'CuDNNLSTM'] and len(weights) == 3:
        # determine if we're loading a CuDNNLSTM layer
        # from the number of bias weights:
        # CuDNNLSTM has (units * 8) weights; while LSTM has (units * 4)
        # if there's no bias weight in the file, skip this conversion
        units = weights[1].shape[0]
        bias_shape = weights[2].shape
        n_gates = 4

        if bias_shape == (2 * units * n_gates,):
            source = 'CuDNNLSTM'
        elif bias_shape == (units * n_gates,):
            source = 'LSTM'
        else:
            raise ValueError('Invalid bias shape: ' + str(bias_shape))

        def convert_weights(weights, from_cudnn=True):
            # transpose (and reshape) input and recurrent kernels
            kernels = transform_kernels(weights[0],
                                        transpose_input(from_cudnn),
                                        n_gates)
            recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
            if from_cudnn:
                # merge input and recurrent biases into a single set
                biases = np.sum(np.split(weights[2], 2, axis=0), axis=0)
            else:
                # Split single set of biases evenly to two sets. The way of
                # splitting doesn't matter as long as the two sets sum is kept.
                biases = np.tile(0.5 * weights[2], 2)
            return [kernels, recurrent_kernels, biases]

        if source != target_class:
            weights = convert_weights(weights, from_cudnn=source == 'CuDNNLSTM')

    # convert the weights between CuDNNGRU and GRU(reset_after=True)
    if target_class in ['GRU', 'CuDNNGRU'] and len(weights) == 3:
        # We can determine the source of the weights from the shape of the bias.
        # If there is no bias we skip the conversion
        # since CuDNNGRU always has biases.

        units = weights[1].shape[0]
        bias_shape = weights[2].shape
        n_gates = 3

        def convert_weights(weights, from_cudnn=True):
            kernels = transform_kernels(weights[0],
                                        transpose_input(from_cudnn),
                                        n_gates)
            recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
            biases = np.array(weights[2]).reshape((2, -1) if from_cudnn else -1)
            return [kernels, recurrent_kernels, biases]

        if bias_shape == (2 * units * n_gates,):
            source = 'CuDNNGRU'
        elif bias_shape == (2, units * n_gates):
            source = 'GRU(reset_after=True)'
        elif bias_shape == (units * n_gates,):
            source = 'GRU(reset_after=False)'
        else:
            raise ValueError('Invalid bias shape: ' + str(bias_shape))

        if target_class == 'CuDNNGRU':
            target = 'CuDNNGRU'
        elif layer.reset_after:
            target = 'GRU(reset_after=True)'
        else:
            target = 'GRU(reset_after=False)'

        # only convert between different types
        if source != target:
            types = (source, target)
            if 'GRU(reset_after=False)' in types:
                raise ValueError('%s is not compatible with %s' % types)
            if source == 'CuDNNGRU':
                weights = convert_weights(weights, from_cudnn=True)
            elif source == 'GRU(reset_after=True)':
                weights = convert_weights(weights, from_cudnn=False)

    return weights


def preprocess_weights_for_loading(layer, weights,
                                   original_keras_version=None,
                                   original_backend=None,
                                   reshape=False):
    """Converts layers weights from Keras 1 format to Keras 2.

    # Arguments
        layer: Layer instance.
        weights: List of weights values (Numpy arrays).
        original_keras_version: Keras version for the weights, as a string.
        original_backend: Keras backend the weights were trained with,
            as a string.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Returns
        A list of weights values (Numpy arrays).
    """
    def convert_nested_bidirectional(weights):
        """Converts layers nested in `Bidirectional` wrapper.

        # Arguments
            weights: List of weights values (Numpy arrays).
        # Returns
            A list of weights values (Numpy arrays).
        """
        num_weights_per_layer = len(weights) // 2
        forward_weights = preprocess_weights_for_loading(
            layer.forward_layer,
            weights[:num_weights_per_layer],
            original_keras_version,
            original_backend)
        backward_weights = preprocess_weights_for_loading(
            layer.backward_layer,
            weights[num_weights_per_layer:],
            original_keras_version,
            original_backend)
        return forward_weights + backward_weights

    def convert_nested_time_distributed(weights):
        """Converts layers nested in `TimeDistributed` wrapper.

        # Arguments
            weights: List of weights values (Numpy arrays).
        # Returns
            A list of weights values (Numpy arrays).
        """
        return preprocess_weights_for_loading(
            layer.layer, weights, original_keras_version, original_backend)

    def convert_nested_model(weights):
        """Converts layers nested in `Model` or `Sequential`.

        # Arguments
            weights: List of weights values (Numpy arrays).
        # Returns
            A list of weights values (Numpy arrays).
        """
        new_weights = []
        # trainable weights
        for sublayer in layer.layers:
            num_weights = len(sublayer.trainable_weights)
            if num_weights > 0:
                new_weights.extend(preprocess_weights_for_loading(
                    layer=sublayer,
                    weights=weights[:num_weights],
                    original_keras_version=original_keras_version,
                    original_backend=original_backend))
                weights = weights[num_weights:]

        # non-trainable weights
        for sublayer in layer.layers:
            ref_ids = [id(w) for w in sublayer.trainable_weights]
            num_weights = len([l for l in sublayer.weights
                               if id(l) not in ref_ids])
            if num_weights > 0:
                new_weights.extend(preprocess_weights_for_loading(
                    layer=sublayer,
                    weights=weights[:num_weights],
                    original_keras_version=original_keras_version,
                    original_backend=original_backend))
                weights = weights[num_weights:]
        return new_weights

    # Convert layers nested in Bidirectional/TimeDistributed/Model/Sequential.
    # Both transformation should be ran for both Keras 1->2 conversion
    # and for conversion of CuDNN layers.
    if layer.__class__.__name__ == 'Bidirectional':
        weights = convert_nested_bidirectional(weights)
    if layer.__class__.__name__ == 'TimeDistributed':
        weights = convert_nested_time_distributed(weights)
    elif layer.__class__.__name__ in ['Model', 'Sequential']:
        weights = convert_nested_model(weights)

    if original_keras_version == '1':
        if layer.__class__.__name__ == 'TimeDistributed':
            weights = preprocess_weights_for_loading(layer.layer,
                                                     weights,
                                                     original_keras_version,
                                                     original_backend)

        if layer.__class__.__name__ == 'Conv1D':
            shape = weights[0].shape
            # Handle Keras 1.1 format
            if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
                # Legacy shape:
                # (filters, input_dim, filter_length, 1)
                assert (shape[0] == layer.filters and
                        shape[2:] == (layer.kernel_size[0], 1))
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
            weights[0] = weights[0][:, 0, :, :]

        if layer.__class__.__name__ == 'Conv2D':
            if layer.data_format == 'channels_first':
                # old: (filters, stack_size, kernel_rows, kernel_cols)
                # new: (kernel_rows, kernel_cols, stack_size, filters)
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

        if layer.__class__.__name__ == 'Conv2DTranspose':
            if layer.data_format == 'channels_last':
                # old: (kernel_rows, kernel_cols, stack_size, filters)
                # new: (kernel_rows, kernel_cols, filters, stack_size)
                weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
            if layer.data_format == 'channels_first':
                # old: (filters, stack_size, kernel_rows, kernel_cols)
                # new: (kernel_rows, kernel_cols, filters, stack_size)
                weights[0] = np.transpose(weights[0], (2, 3, 0, 1))

        if layer.__class__.__name__ == 'Conv3D':
            if layer.data_format == 'channels_first':
                # old: (filters, stack_size, ...)
                # new: (..., stack_size, filters)
                weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))

        if layer.__class__.__name__ == 'GRU':
            if len(weights) == 9:
                kernel = np.concatenate([weights[0],
                                         weights[3],
                                         weights[6]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[4],
                                                   weights[7]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[5],
                                       weights[8]], axis=-1)
                weights = [kernel, recurrent_kernel, bias]

        if layer.__class__.__name__ == 'LSTM':
            if len(weights) == 12:
                # old: i, c, f, o
                # new: i, f, c, o
                kernel = np.concatenate([weights[0],
                                         weights[6],
                                         weights[3],
                                         weights[9]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[7],
                                                   weights[4],
                                                   weights[10]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[8],
                                       weights[5],
                                       weights[11]], axis=-1)
                weights = [kernel, recurrent_kernel, bias]

        if layer.__class__.__name__ == 'ConvLSTM2D':
            if len(weights) == 12:
                kernel = np.concatenate([weights[0],
                                         weights[6],
                                         weights[3],
                                         weights[9]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[7],
                                                   weights[4],
                                                   weights[10]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[8],
                                       weights[5],
                                       weights[11]], axis=-1)
                if layer.data_format == 'channels_first':
                    # old: (filters, stack_size, kernel_rows, kernel_cols)
                    # new: (kernel_rows, kernel_cols, stack_size, filters)
                    kernel = np.transpose(kernel, (2, 3, 1, 0))
                    recurrent_kernel = np.transpose(recurrent_kernel,
                                                    (2, 3, 1, 0))
                weights = [kernel, recurrent_kernel, bias]

    conv_layers = ['Conv1D',
                   'Conv2D',
                   'Conv3D',
                   'Conv2DTranspose',
                   'ConvLSTM2D']
    if layer.__class__.__name__ in conv_layers:
        layer_weights_shape = keras.backend.int_shape(layer.weights[0])
        if _need_convert_kernel(original_backend):
            weights[0] = convert_kernel(weights[0])
            if layer.__class__.__name__ == 'ConvLSTM2D':
                weights[1] = convert_kernel(weights[1])
        if reshape and layer_weights_shape != weights[0].shape:
            if weights[0].size != np.prod(layer_weights_shape):
                raise ValueError('Weights must be of equal size to ' +
                                 'apply a reshape operation. ' +
                                 'Layer ' + layer.name +
                                 '\'s weights have shape ' +
                                 str(layer_weights_shape) + ' and size ' +
                                 str(np.prod(layer_weights_shape)) + '. ' +
                                 'The weights for loading have shape ' +
                                 str(weights[0].shape) + ' and size ' +
                                 str(weights[0].size) + '. ')
            weights[0] = np.reshape(weights[0], layer_weights_shape)
        elif layer_weights_shape != weights[0].shape:
            weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
            if layer.__class__.__name__ == 'ConvLSTM2D':
                weights[1] = np.transpose(weights[1], (3, 2, 0, 1))

    # convert CuDNN layers
    weights = _convert_rnn_weights(layer, weights)

    return weights


# ----------------------------------------------------------- #
#   载入权重文件(用来解决tf2.0中缺少skip_mismatch的功能)
#   默认的是通过name来导入，skip_mismatch=True
#   # 如果需要通过拓扑结构来导入权重，直接使用model.load_weights
# ----------------------------------------------------------- #
def load_weights_my(model, weights_path, by_name=True, skip_mismatch=True):
    if h5py is None:
        raise ImportError('`load_weights` 需要 h5py.')
    if not isinstance(by_name, bool):
        raise ValueError('by_name请使用bool类型参数')
    if not by_name:
        raise ValueError('请使用name进行权重加载')

    f = h5py.File(weights_path, mode='r')
    # 网络层
    layers = model.layers

    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    if by_name and skip_mismatch:
        if 'keras_version' in f.attrs:
            original_keras_version = f.attrs['keras_version'].decode('utf8')
        else:
            original_keras_version = '1'
        if 'backend' in f.attrs:
            original_backend = f.attrs['backend'].decode('utf8')
        else:
            original_backend = None

        layer_names = [name.decode('utf-8') for name in f.attrs['layer_names']]

        # Reverse index of layer name to list of layers with name.
        index = {}
        for layer in layers:
            if layer.name:
                index.setdefault(layer.name, []).append(layer)

        weight_value_tuples = []

        for k, name in enumerate(layer_names):
            gro = f[name]
            # 查找weight_names
            weight_names = [name.decode('utf-8') for name in gro.attrs['weight_names']]
            # 将weight_value转化为array
            weight_values = [np.asarray(gro[name]) for name in weight_names]

            for layer in index.get(name, []):
                symbolic_weights = layer.weights
                weight_values = preprocess_weights_for_loading(
                    layer,
                    weight_values,
                    original_keras_version,
                    original_backend,
                    reshape=False)
                if len(weight_values) != len(symbolic_weights):
                    if skip_mismatch:
                        warnings.warn('Skipping loading of weights for '
                                      'layer {}'.format(layer.name) + ' due to mismatch '
                                      'in number of weights ({} vs {}).'.format(
                                        len(symbolic_weights), len(weight_values)))
                        continue
                    else:
                        raise ValueError('Layer #' + str(k) +
                                         ' (named "' + layer.name +
                                         '") expects ' +
                                         str(len(symbolic_weights)) +
                                         ' weight(s), but the saved weights' +
                                         ' have ' + str(len(weight_values)) +
                                         ' element(s).')
                # Set values.
                for i in range(len(weight_values)):
                    symbolic_shape = keras.backend.int_shape(symbolic_weights[i])
                    if symbolic_shape != weight_values[i].shape:
                        if skip_mismatch:
                            warnings.warn('Skipping loading of weights for '
                                          'layer {}'.format(layer.name) + ' due to '
                                          'mismatch in shape ({} vs {}).'.format(
                                            symbolic_weights[i].shape,
                                            weight_values[i].shape))
                            continue
                        else:
                            raise ValueError('Layer #' + str(k) +
                                             ' (named "' + layer.name +
                                             '"), weight ' +
                                             str(symbolic_weights[i]) +
                                             ' has shape {}'.format(symbolic_shape) +
                                             ', but the saved weight has shape ' +
                                             str(weight_values[i].shape) + '.')
                    else:
                        weight_value_tuples.append((symbolic_weights[i],
                                                    weight_values[i]))

        keras.backend.batch_set_value(weight_value_tuples)

    if hasattr(f, 'close'):
        f.close()
    elif hasattr(f.file, 'close'):
        f.file.close()

    return model


# ------------------------------- #
#   h5py读取.h5文件
#   f = h5py.file()
#   f.attrs 根据attrs找到键值
#   f.keys  直接得到键值
# ------------------------------- #
def read_h5(weights_path):
    f = h5py.File(weights_path, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    # group
    # 根据attr查找key
    attr_names = list(f.attrs)
    print('文件中有这些属性:{}'.format(attr_names))
    layer_names = list(f.attrs['layer_names'])
    print('层的名称为：{}'.format(layer_names))
    print('weighs文件中总共包含{}层'.format(len(layer_names)))
    f.file.close()


if __name__ == '__main__':
    # read_h5('./yolo_weights.h5')

    # 测试my_load_weights
    from net.dark53 import yolo3_body

    inputs = keras.Input(shape=(416, 416, 3))
    yolo3 = yolo3_body(inputs, 3, 80)
    load_weights_my(yolo3, './yolo_weights.h5', by_name=True, skip_mismatch=True)
