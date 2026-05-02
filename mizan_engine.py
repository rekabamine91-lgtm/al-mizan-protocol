import numpy as np

class AlMizanEngine:
    def __init__(self):
        self.version = "0.1.0"

    def apply_qist_balance(self, values):
        """
        تطبيق معادلة القسط لموازنة البيانات.
        """
        if not isinstance(values, (list, np.ndarray)):
            raise ValueError("Values must be a list or numpy array")
        
        arr = np.array(values)
        mean_val = np.mean(arr)
        # موازنة القيم التي تتجاوز الانحراف المعياري
        balanced = np.where(arr > mean_val * 2, mean_val * 1.5, arr)
        return balanced.tolist()
