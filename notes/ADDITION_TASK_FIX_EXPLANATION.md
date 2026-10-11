# Addition Task: Design Flaw and Fix

## The Problem

The original addition task had **boundaries inserted in a structurally naive way**, making it impossible to fairly test whether boundaries affect carry propagation.

---

## Concrete Example

Adding two 4-digit numbers with boundary interval k=2:
- **Number 1:** 1234
- **Number 2:** 5678
- **Correct sum:** 7012

---

## OLD FORMAT (Original, Broken)

### How It Worked

The code concatenated both numbers with the operator, then inserted boundaries uniformly:

```python
combined = list(a_str) + ["+"] + list(b_str)
# Result: ['1','2','3','4','+','5','6','7','8']

# Insert "|" every k=2 positions across the ENTIRE combined string
for i, d in enumerate(combined):
    if (i + 1) % 2 == 0:
        insert "|"
```

### Input String Generated

```
12|34|+5|67|8
```

### Tokenization (What the Model Sees)

```
Position:  1   2   3   4   5   6   7   8   9  10  11  12  13
Token:    ['1','2','|','3','4','|','+','5','|','6','7','|','8']
```

### The Problem

1. **"+" is buried:** The operator appears at position 7, scattered among boundaries
2. **Numbers are broken apart:** 
   - First number looks like: `12|34`
   - Second number looks like: `5|67|8` (WRONG! Should be `5678`)
3. **Structure is completely broken:** The model can't even parse which digits belong to which number
4. **Result:** TAR ≈ 0, but for the WRONG reason (parsing failure, not carry blocking)

### Example
```
OLD FORMAT:  12|34|+5|67|8
             ^^     ^ ^^   ^
             │      │ │└─ This looks like a separate number
             │      │ └── Where did 5678 go?
             │      └───── Operator buried here
             └──────────── Did this belong to first or second number?
```

---

## NEW FORMAT (Structured, Fair)

### How It Works

Insert boundaries **within each operand separately**, then combine:

```python
a_bounded = insert_boundaries_in_number(a_str, k=2)
# "1234" → "12|34"

b_bounded = insert_boundaries_in_number(b_str, k=2)
# "5678" → "56|78"

input_str = f"{a_bounded}+{b_bounded}"
# "12|34+56|78"
```

### Input String Generated

```
12|34+56|78
```

### Tokenization (What the Model Sees)

```
Position:  1   2   3   4   5   6   7   8   9  10  11
Token:    ['1','2','|','3','4','+','5','6','|','7','8']
```

### The Advantages

1. **"+" is visible:** Clear operator at position 5
2. **Numbers are kept together:** 
   - First number: `12|34` (clearly the first operand)
   - Second number: `56|78` (clearly the second operand)
3. **Structure is preserved:** The model can parse the problem correctly
4. **Fair test:** Now we can test the ACTUAL hypothesis (carry blocking), not parsing ability

### Example
```
NEW FORMAT:  12|34+56|78
             ^^^^^ ^ ^^^^
             │     │ └─── Second number with boundaries
             │     └────── Clear operator
             └──────────── First number with boundaries
```

---

## Why This Matters

### What the Old Format Tests
- Can the model handle **malformed, confusing input**?
- Result: TAR ≈ 0 (partly because structure is broken, partly because of carries)
- **Not a fair test of the boundary hypothesis**

### What the New Format Tests
- **Do boundaries block carry propagation** in addition?
- Result: TAR might still be low (if carries are truly blocked)
- OR: TAR might be higher (if the carry-blocking hypothesis is wrong)
- **Fair test of the theoretical hypothesis**

---

## No Extra Tokens Added

**Important:** We did NOT add spaces or parentheses around the operator. The only difference is:

| Aspect | OLD | NEW |
|--------|-----|-----|
| Boundary placement | Uniform across `a+b` | Separate within `a` and `b` |
| Extra tokens | None | None |
| Structure | Broken | Preserved |
| Fair hypothesis test | ✗ No | ✓ Yes |

---

## Summary

| Format | Input | Problem | Tests What |
|--------|-------|---------|-----------|
| **Original (Old)** | `12\|34\|+5\|67\|8` | Numbers are split incorrectly, operator buried | Model's ability to parse malformed input ❌ |
| **Fixed (New)** | `12\|34+56\|78` | None — structure is clear | Whether boundaries block carry propagation ✓ |

---

## Next Steps

We've created a corrected version:
- **Script:** `run_addition_structured_boundaries.py`
- **Batch job:** `run_addition_structured_boundaries_batch.sh`
- **Submitted job ID:** 36006586
- **Results will be saved to:** `addition_grid_structured_boundaries.csv`

This allows us to fairly test the carry-direction hypothesis for the addition task.
