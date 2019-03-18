import torch

def calc_log_reg_hessian(model, data):
  with torch.no_grad():
    out = model(data)
    return data.matmul(data) * out

def calc_log_reg_hessian_inverse(model, data):
  hessian = calc_log_reg_hessian(model, data)
  return torch.inverse(hessian)

def calc_log_reg_hvp_inverse(model, data, grads):
  hessian_inv = calc_log_reg_hessian_inverse(model, data)
  return grads.matmul(hessian_inv)

def calc_log_reg_grad(model, data, target):
  with torch.no_grad():
    out = model(data)
    return data * (out - target)

def calc_s_tests(calc_hvp_inverse, calc_grad, test_data, test_target):
  grads = calc_grad(test_data, test_target)
  return calc_hvp_inverse(grads)

def calc_influences(calc_grad, s_tests, data):
  grads = calc_grad(data)
  return - grads.matmul(s_tests.t())
