from mizan_engine import MizanValue

def run_test():
    print("--- Al-Mizan Protocol v1.2.2 Testing ---")
    
    # تعريف مدخلات الشبكة (خلايا عصبية بمدخلات وموازين)
    x1 = MizanValue(2.0, label='x1')
    x2 = MizanValue(0.0, label='x2')
    
    w1 = MizanValue(-3.0, label='w1')
    w2 = MizanValue(1.0, label='w2')
    
    b = MizanValue(6.88, label='b')
    
    # العملية الحسابية: (x1*w1 + x2*w2) + b
    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    
    # تشغيل البروتوكول
    o.apply_mizan_protocol()
    
    print(f"Output Score: {o.data:.4f}")
    print(f"Gradient of w1 (with Mizan): {w1.grad:.4f}")
    print(f"Tyranny Check: {'Safe' if w1.tyranny_count == 0 else 'Regulated'}")

if __name__ == "__main__":
    run_test()
