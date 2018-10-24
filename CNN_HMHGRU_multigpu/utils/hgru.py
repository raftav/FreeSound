import collections
import tensorflow as tf
from utils import binary_ops

from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope as vs


##########################################
# Tuple that will be passed as cell state
##########################################
_HardGatedStateTuple = collections.namedtuple("HardGatedStateTuple", ("h", "z"))

##########################################
# Tuple that will be passed as cell output
##########################################
_HardGatedOutputTuple = collections.namedtuple("HardGatedOutputTuple", ("h", "z", "z_tilda"))

class HardGatedStateTuple(_HardGatedStateTuple):
  """Tuple used by HardGated Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(h, z)`, in that order.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, z ) = self
    return h.dtype

class HardGatedOutputTuple(_HardGatedOutputTuple):
  """Tuple used by HardGated Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(h, z)`, in that order.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, z , z_tilda) = self
    return h.dtype

#########################
# HGRU Cell definition
#########################  
class HGRUCell(tf.contrib.rnn.RNNCell):

  slope=None

  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def state_size(self):
    return HardGatedStateTuple(self._num_units, 1 )

  @property
  def output_size(self):
    return HardGatedOutputTuple(self._num_units, 1 ,1 )

  def _norm(self,inp, scope , norm_gain=1.0, norm_shift=0.0):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(norm_gain)
    beta_init = init_ops.constant_initializer(norm_shift)

    with vs.variable_scope(scope):
      vs.get_variable("gamma", shape=shape, initializer=gamma_init)
      vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def __call__(self, inputs, state, scope=None):

    h_bottom, z_bottom, h_top_prev = inputs
    h_prev, z_prev  = state

    with vs.variable_scope(scope or type(self).__name__):

      ################
      # UPDATE MODULE
      ################

      # Matrix U_l^l
      U_curr = vs.get_variable("U_curr", [h_prev.get_shape()[1], self._num_units], dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
      # Matrix U_{l-1}^l
      U_bottom = vs.get_variable("U_bottom", [h_bottom.get_shape()[1], self._num_units], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      # Matrix R_{l-1}^l
      R_bottom = vs.get_variable("R_bottom", [h_bottom.get_shape()[1], self._num_units], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      #Matrix R_l^l
      R_curr = vs.get_variable("R_curr", [h_prev.get_shape()[1], self._num_units], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      with tf.name_scope('reset_input') as input_scope:
        reset_input = tf.add(tf.matmul(h_bottom,R_bottom),tf.matmul(h_prev,R_curr),name=input_scope)


      reset_input = self._norm(reset_input,"reset",norm_shift=1.0)

      with tf.name_scope('reset_input_rescaled') as input_scope:
        reset_input = tf.identity(reset_input,name=input_scope)

      # reset gate as in GRU
      reset_gate = tf.sigmoid( reset_input )

      with tf.name_scope('update_input') as input_scope:
        u_input = tf.add(tf.matmul(h_bottom,U_bottom),tf.matmul(tf.multiply(reset_gate,h_prev),U_curr),name=input_scope)

      u_input = self._norm(u_input,"update")
      with tf.name_scope('update_input_rescaled') as input_scope:
        u_input=tf.identity(u_input,name=input_scope)

      # u_t^l : essentially a GRU
      u_t = tf.tanh( u_input )

      ################
      # FLUSH MODULE
      ################

      # Matrix W_{l+1}^l
      W_top = vs.get_variable("W_top", [h_top_prev.get_shape()[1], self._num_units], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      # Matrix W_{l-1}^l
      W_bottom = vs.get_variable("W_bottom", [h_bottom.get_shape()[1], self._num_units], dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      with tf.name_scope('flush_input') as input_scope:
        flush_input = tf.add(tf.matmul(h_top_prev,W_top),tf.matmul(h_bottom,W_bottom),name=input_scope)

      flush_input = self._norm(flush_input,"flush")
      with tf.name_scope('flush_input_rescaled') as input_scope:
        flush_input = tf.identity(flush_input,name=input_scope)

      # f_t^l : the flush module
      f_t = tf.tanh( flush_input )

      ##################
      # BINARY UNIT
      ##################

      # Matrix V_l^l
      V_curr = vs.get_variable("V_curr", [h_prev.get_shape()[1], 1 ], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

      V_bottom = vs.get_variable("V_bottom", [h_bottom.get_shape()[1], 1 ], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1)) 

      bias_z = vs.get_variable("bias_z", shape=[1],dtype=tf.float32,
      	                       initializer=tf.ones_initializer())

      with tf.name_scope('gates_input') as input_scope:
        z_tilda_input = tf.add(tf.matmul(h_prev,V_curr),tf.matmul(h_bottom,V_bottom),name=input_scope)
        z_tilda_input = tf.add(z_tilda_input,bias_z)

      z_tilda_input = z_bottom * z_tilda_input

      with tf.name_scope('gates_input_rescaled') as input_scope:
        z_tilda_logits = tf.identity(z_tilda_input,name=input_scope)

      z_new =  binary_ops.binary_wrapper(z_tilda_logits,
                             pass_through=False, 
                             stochastic_tensor=tf.constant(False),
                             slope_tensor=self.slope)

      #################
      # HIDDEN LAYER
      #################

      h_new = (tf.ones_like(z_new) - z_new) * ( (tf.ones_like(z_bottom) - z_bottom) * h_prev + z_bottom * u_t ) + z_new * f_t

    state_new = HardGatedStateTuple(h_new, z_new)
    output_new = HardGatedOutputTuple(h_new, z_new,z_tilda_logits)

    return output_new, state_new


#########################
# Vertical Stack of HGRU Cells
#########################

class MultiHGRUCell(tf.contrib.rnn.RNNCell):
  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of HGRNNCells.

    Args:
      cells: list of HGRNNCells that will be composed in this order.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiHmRNNCell.")
    self._cells = cells

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    return tuple(cell.output_size for cell in self._cells)


  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    assert len(state) == len(self._cells)
    with vs.variable_scope(scope or type(self).__name__):  # "MultiHmRNNCell"

      # assign h_prev_top considering the special case of only one hidden layer
      if len(self._cells) > 1:
        h_prev_top = state[1].h
      else:
        h_prev_top = tf.zeros(tf.shape(state[0].h))

      # set inputs. 
      # here the gates at the first layer are set to 1.
      #input_boundaries = tf.get_variable('input_z',shape=[inputs.get_shape()[1], 1],initializer=tf.ones_initializer())
      #current_input = inputs, input_boundaries, h_prev_top
      current_input = inputs, tf.ones([tf.shape(inputs)[0], 1]), h_prev_top

      new_h_list = []
      new_states = []
       
      # Go through each cell in the different layers, going bottom to top
      # place cells in different devices

      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):

          new_h, new_state = cell(current_input, state[i])

          # Set up the inputs for the next cell.
          if i < len(self._cells) - 2:
            # Next cell is not the top one.
            h_prev_top = state[i+2].h
          else:
            # The next cell is the top one, so give it zeros for its h_prev_top input.
            h_prev_top = tf.zeros(tf.shape(state[i].h))

          # update input  
          current_input = new_state.h, new_state.z, h_prev_top  # h_bottom, z_bottom, h_prev_top
          
          #save outputs and states
          new_h_list.append(new_h)
          new_states.append(new_state)

      # return a tuple with the activation of all the hidden layers
      output = tuple(new_h_list)

    return output, tuple(new_states)




