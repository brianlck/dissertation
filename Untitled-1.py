
    def clipped_grad(self, x, t):
        grad_log_init: Tensor = self.grad_initial_log(x)
        grad_log_target: Tensor = self.grad_target_log(x)
        
        tmp = torch.clip(grad_log_target.norm(dim=1)[:, None], min=0, max=1e2)
        grad_log_target = torch.nn.functional.normalize(grad_log_target) * torch.clip(tmp, min=0, max=1e2)

        tmp = torch.clip(grad_log_init.norm(dim=1)[:, None], min=0, max=1e2)
        grad_log_init = torch.nn.functional.normalize(grad_log_init) * torch.clip(tmp, min=0, max=1e2)
        
        return (1-t) * grad_log_init + t * grad_log_target