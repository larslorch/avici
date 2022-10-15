import gc
import re
import jax
import jax.interpreters
import jax.numpy as jnp
import numpy as np
import pandas as pd

def natural_keys(text):
  """Key function for natural sort of strings."""
  atoi = lambda text: int(text) if text.isdigit() else text
  return [atoi(c) for c in re.split(r'(\d+)', text)]


def pandas_table(col_dict, ascii=True):
  """Make pandas dataframe built from dict of cols. Useful for table diplay."""
  df = pd.concat([pd.DataFrame({k: v}) for k, v in col_dict.items()], axis=1)
  if ascii:
    return df.to_string(index=False)
  else:
    raise NotImplementedError("not in jupyter notebook; import this otherwise: `from IPython import display`")
    # return display.HTML(df.to_html(index=False))


# Memory Reporting and Management
def object_memory_usage(output='asciitable'):
  """Find all tracked DeviceValues and calculate their device memory in use."""
  DeletedBuffer = jax.interpreters.xla.DeletedBuffer
  liveset = {}
  gc.collect()  # necessary?
  # dvals = (x for x in gc.get_objects() if isinstance(x, jax.xla.DeviceValue))
  dvals = (x for x in gc.get_objects() if (
    isinstance(x, jax.xla.DeviceArray) or
    isinstance(x, jax.xla._DeviceArray) or
    # isinstance(x, jax.xla.DeviceConstant) or
    isinstance(x, jax.pxla.ShardedDeviceArray)
  ))
  for dv in dvals:
    # # DeviceConstants are lazy and use no memory.
    # if jax.xla.is_device_constant(dv):
    #   continue
    # Don't count memory that's already been released.
    if hasattr(dv, 'device_buffer') and (
        isinstance(dv.device_buffer, DeletedBuffer)
        or dv.device_buffer.is_deleted()):
      continue
    # Only count shared device_buffer usage once.
    if isinstance(dv, jax.pxla.ShardedDeviceArray):
      if dv.device_buffers:
        for db in dv.device_buffers:
          if isinstance(db, DeletedBuffer) or db.is_deleted():
            continue
          liveset[id(db)] = (db, np.prod(dv.aval.shape[1:]) *
                            dv.aval.dtype.itemsize)
    elif isinstance(dv, jax.xla.DeviceArray):
      db = dv.device_buffer
      liveset[id(db)] = (db, np.prod(dv.aval.shape) * dv.aval.dtype.itemsize)
  results = list(liveset.values())

  # format output as HTML table for colab/jupyter.
  if output in ('asciitable', 'table'):
    counts, mem = {}, {}
    for db, sz in results:
      key = '%s%d' % (db.platform(), db.device().id)
      counts[key] = counts.get(key, 0) + 1
      mem[key] = mem.get(key, 0) + sz
    # calculate totals
    counts['total'] = np.sum(list(counts.values()))
    mem['total'] = np.sum(list(mem.values()))
    devkeys = sorted(list(counts.keys()), key=natural_keys)
    return pandas_table({'device': devkeys,
                         'count': [counts[k] for k in devkeys],
                         'memory': ['{:,}'.format(mem[k]) for k in devkeys]
                         }, ascii=(output == 'asciitable'))

  # just return total number of bytes used
  elif output == 'simple':
    return np.sum([r[1] for r in results])
  # return list of dedup'd live device buffers for debugging purposes
  elif output == 'buffers':
    return [r[0] for r in results]
  else:
    raise ValueError('output must be one of table, simple, or buffers.')


def reset_device_memory(delete_objs=True):
  """Free all tracked DeviceArray memory and delete objects.

  Args:
    delete_objs: bool: whether to delete all live DeviceValues or just free.

  Returns:
    number of DeviceArrays that were manually freed.
  """
  DeletedBuffer = jax.interpreters.xla.DeletedBuffer
  dvals = (x for x in gc.get_objects() if isinstance(x, jax.xla.DeviceValue))
  n_deleted = 0
  for dv in dvals:
    if not jax.xla.is_device_constant(dv):
      if (hasattr(dv, 'device_buffer') and
          isinstance(dv.device_buffer, DeletedBuffer)):
        pass
      elif hasattr(dv, 'device_buffers') and dv.device_buffers is None:
        pass
      else:
        dv.delete()
        n_deleted += 1
    if delete_objs:
      del dv
  del dvals
  gc.collect()
  return n_deleted

# Warning: I wrote this 4 months ago, it may have bitrot as internals evolved,
# it seems to run as of Mar 11 2020 though.
def clear_jax_caches():
  """Utility to clear all the function caches in jax."""
  # main jit/pmap lu wrapped function caches - have to grab from closures
  jax.xla._xla_callable.__closure__[1].cell_contents.clear()
  jax.pxla.parallel_callable.__closure__[1].cell_contents.clear()
  # primitive callable caches
  jax.xla.xla_primitive_callable.cache_clear()
  jax.xla.primitive_computation.cache_clear()
  # jaxpr caches for control flow and reductions
  jax.lax.lax_control_flow._initial_style_jaxpr.cache_clear()
  jax.lax.lax_control_flow._fori_body_fun.cache_clear()
  jax.lax.lax._reduction_jaxpr.cache_clear()
  # these are trivial and only included for completeness sake
  jax.lax.lax.broadcast_shapes.cache_clear()
  jax.xla.xb.get_backend.cache_clear()
  jax.xla.xb.dtype_to_etype.cache_clear()
  jax.xla.xb.supported_numpy_dtypes.cache_clear()