import numpy as np


class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params.values():
            p.data = p.data - p.grad * self.lr


class SGDMomentum:
    def __init__(self, params, lr, beta1=0.9):
        self.params = params
        self.lr = lr
        self.beta1 = beta1

        # First-Moment Dictionaries
        self.t = 0
        self.m = {}
        for name, p in params.items():
            self.m[name] = np.zeros_like(p.grad)

    def step(self):
        self.t += 1
        for name, p in self.params.items():
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * p.grad
            p.data -= self.lr * self.m[name]


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-6):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = {}
        self.v = {}
        for name, p in params.items():
            self.m[name] = np.zeros_like(params[name].grad)
            self.v[name] = np.zeros_like(params[name].grad)

    def step(self):
        self.t += 1

        for name, p in self.params.items():
            # Abbreviate
            data, g = p.data, p.grad
            m, v = self.m[name], self.v[name]
            b1, b2, t, lr, eps = self.beta1, self.beta2, self.t, self.lr, self.eps

            # Compute Moving Averages
            m = m * b1 + g * (1 - b1)

            v = v * b2 + g**2 * (1 - b2)

            # Perform corrections
            # Step 1: b1 * 0 + (1 - b1) * m1 -> divide by (1 - b1)
            # Step 2: b1 * 0 + b1 * (1 - b1) * m1 + (1 - b1) * m2
            # ^Coefficients are b1 * (1 - b1) + (1 - b1) = (1 - b1)*(1 + b1)
            # Step 3: b1 * 0 + b1 * b1 * (1 - b1) * m1 + b1 * (1 - b1) * m2 + (1 - b1) * m3 ..
            # Pattern is: (1 - b1)*(1 + b1 + b1**2 + ... + b1**(t-1))
            # (1 + b1 + b1**2 + ... + b1**(t-1)) - (b1 + b1**2 + ... + b1**t)
            # (1 - b1 ** t)
            m_tilde = m / (1 - b1**t)
            v_tilde = v / (1 - b2**t)

            data = data - lr * (m_tilde / (np.sqrt(v_tilde) + eps))

            # Update moving averages and parameter data
            self.m[name], self.v[name] = m, v
            p.data = data
