import torch

def calc_log_reg_hessian(model, data):
  with torch.no_grad():
    out = model(data)
    return (data.t() * out).matmul(data) / len(data)

def calc_log_reg_hessian_inverse(model, data, reg=0.01):
  hessian = calc_log_reg_hessian(model, data)
  return torch.inverse(hessian + torch.eye(len(hessian)) * reg)

def calc_log_reg_hvp_inverse(model, data, grads, reg=0.01):
  hessian_inv = calc_log_reg_hessian_inverse(model, data, reg=reg)
  return grads.matmul(hessian_inv)

def calc_log_reg_grad(model, data, target):
  with torch.no_grad():
    out = model(data)
    return data * (out - target).unsqueeze(1)

def calc_log_reg_dkl_grad(model, data, target, protected_col_idx=-1):
  protected_rows = data[:, protected_col_idx] == 0
  unprotected_rows = data[:, protected_col_idx] == 1
  with torch.no_grad():
    out = model(data)
    p = torch.mean(out[protected_rows])
    q = torch.mean(out[unprotected_rows])
    g_p = torch.mean(calc_log_reg_grad(model, data[protected_rows], target[protected_rows]), 0)
    g_q = torch.mean(calc_log_reg_grad(model, data[unprotected_rows], target[unprotected_rows]), 0)
    y_1 = g_p * torch.log(p / q) + q / p**2 * (g_q * p - g_p * q)
    y_0 = - g_p * torch.log((1 - p) / (1 - q)) + (1 - q) / (1 - p)**2 * (- g_q * p + g_p * q)
    return - (y_1 + y_0).unsqueeze(0)

def calc_s_tests(model, calc_hvp_inverse, calc_grad, test_data, test_target, reg=0.01):
  grads = calc_grad(model, test_data, test_target)
  return calc_hvp_inverse(model, grads, reg=reg)

def calc_influences(model, calc_grad, s_tests, data, target):
  grads = calc_grad(model, data, target)
  return - grads.matmul(s_tests.t())
