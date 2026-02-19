
def test_logic(prediction):
    # Determine label (sigmoid output: 0 = Fake, 1 = Real)
    # prediction > 0.5 means closer to 1 (Real)
    is_real = prediction > 0.5
    label = "Real" if is_real else "Fake"
    
    # Adjust confidence for display
    # If Real (p > 0.5), confidence is p * 100
    # If Fake (p <= 0.5), confidence is (1 - p) * 100
    confidence = prediction * 100
    display_confidence = confidence if is_real else (100 - confidence)
    
    return label, display_confidence

print("Verifying Logic Fix:")
# Test Case 1: Real Image (e.g. 0.95) -> Should be "Real", 95%
l1, c1 = test_logic(0.95)
print(f"Pred 0.95: Label='{l1}', Confidence={c1:.2f}% (Expected: Real, 95.00%)")

# Test Case 2: Fake Image (e.g. 0.05) -> Should be "Fake", 95%
l2, c2 = test_logic(0.05)
print(f"Pred 0.05: Label='{l2}', Confidence={c2:.2f}% (Expected: Fake, 95.00%)")

# Test Case 3: Ambiguous (e.g. 0.6) -> Should be "Real", 60%
l3, c3 = test_logic(0.60)
print(f"Pred 0.60: Label='{l3}', Confidence={c3:.2f}% (Expected: Real, 60.00%)")

# Test Case 4: Ambiguous (e.g. 0.4) -> Should be "Fake", 60%
l4, c4 = test_logic(0.40)
print(f"Pred 0.40: Label='{l4}', Confidence={c4:.2f}% (Expected: Fake, 60.00%)")
