# Boundary-Token Task Suite Specification
## Complete Input/Output Examples and Prompts

---

## TIER 1: Position-Local (Each output depends ONLY on current input)

### Task 1.1: Cyclic Character Shift ✓ (Already have)

**Rule:** Replace each character with the next one in alphabet, wrapping Z→A
- A→B, B→C, C→D, ..., Y→Z, Z→A

**Alphabet:** A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

**Input example (N=4, k=2):**
```
YZ|AB
```

**Output example:**
```
ZA|BC
```
(Calculation: Y→Z, Z→A (wraps!), A→B, B→C)

**Prompt (complete):**
```
You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
Rule: Replace every character with the next one in the alphabet.
Mapping: Z → A (wraps around), and A → B.
Constraint: The input contains vertical bars (|). These are structural boundaries. 
DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: YZ|AB
Output: 
```

---

### Task 1.2: Character Substitution Mapping (NEW)

**Rule:** Replace each character with a fixed mapped character
- Fixed mapping: A→X, B→Y, C→Z, D→W, E→P, ...

**Alphabet:** A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

**Mapping:** A↔X, B↔Y, C↔Z, D↔W, E↔P, F↔Q, G↔R, H↔S, I↔T, J↔U, K↔V, L↔M, N↔O (13-letter swap)

**Input example (N=8, k=4):**
```
ABCD|EFGH
```

**Output example:**
```
XYZW|PQRS
```
(Calculation: A→X, B→Y, C→Z, D→W, E→P, F→Q, G→R, H→S)

**Prompt (complete):**
```
You are a deterministic sequence transformer.
Task: Character substitution mapping.
Alphabet: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
Mapping: A↔X, B↔Y, C↔Z, D↔W, E↔P, F↔Q, G↔R, H↔S, I↔T, J↔U, K↔V, L↔M, N↔O
Rule: Replace each character with its mapped character. This is a fixed one-to-one substitution cipher.
Constraint: The input contains vertical bars (|). These are structural boundaries. 
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: ABCD|EFGH
Output: 
```

---

## TIER 2: Single-Previous Dependency (Each output depends on current input + previous input)

### Task 2.1: Running XOR Chain (NEW)

**Rule:** XOR current input with previous input
- First output = first input (no previous input)
- Output[i] = Input[i] XOR Input[i-1]

**Alphabet:** 0, 1 (binary digits)

**Input example (N=8, k=4):**
```
1011|0110
```

**Output example:**
```
1110|1101
```
(Calculation: 
- Pos 0: input=1 → output=1 (first)
- Pos 1: input=0, prev_input=1 → 0 XOR 1 = 1
- Pos 2: input=1, prev_input=0 → 1 XOR 0 = 1
- Pos 3: input=1, prev_input=1 → 1 XOR 1 = 0
- Pos 4 (after |): input=0, prev_input=1 → 0 XOR 1 = 1
- Pos 5: input=1, prev_input=0 → 1 XOR 0 = 1
- Pos 6: input=1, prev_input=1 → 1 XOR 1 = 0
- Pos 7: input=0, prev_input=1 → 0 XOR 1 = 1
)

**Prompt (complete):**
```
You are a deterministic sequence transformer.
Task: Running XOR with previous input.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the previous input digit.
The first digit is copied as-is (no previous input to XOR with).
Each subsequent digit: output[i] = input[i] XOR input[i-1]
Example: Input 1011 → Output 1110 (calculation: 1, 0⊕1=1, 1⊕0=1, 1⊕1=0)
Constraint: The input contains vertical bars (|). These are structural boundaries. 
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 1011|0110
Output: 
```

---

### Task 2.2: Running Sum (Local Window) (NEW)

**Rule:** Add current and previous input, modulo 10
- First output = first input (no previous input to add)
- Output[i] = (Input[i] + Input[i-1]) mod 10

**Alphabet:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (decimal digits)

**Input example (N=8, k=4):**
```
2314|5298
```

**Output example:**
```
2545|9717
```
(Calculation:
- Pos 0: input=2 → output=2 (first)
- Pos 1: (3+2) mod 10 = 5
- Pos 2: (1+3) mod 10 = 4
- Pos 3: (4+1) mod 10 = 5
- Pos 4 (after |): (5+4) mod 10 = 9
- Pos 5: (2+5) mod 10 = 7
- Pos 6: (9+2) mod 10 = 11 → 1
- Pos 7: (8+9) mod 10 = 17 → 7
)

**Prompt (complete):**
```
You are a deterministic sequence transformer.
Task: Running sum with previous input.
Alphabet: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (decimal digits)
Modulus: 10
Rule: At each position, output the sum of the current input digit and the previous input digit, modulo 10.
The first digit is copied as-is (no previous input to add).
Each subsequent digit: output[i] = (input[i] + input[i-1]) mod 10
Example: Input 2314 → Output 2545 (calculation: 2, (3+2)%10=5, (1+3)%10=4, (4+1)%10=5)
Constraint: The input contains vertical bars (|). These are structural boundaries. 
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 2314|5298
Output: 
```

---

## TIER 3: Global-State (Each output depends on all previous inputs)

### Task 3.1: Running Sum Modulo 10 ✓ (Already have as "modsum")

**Rule:** Cumulative sum of digits, modulo 10
- Output[i] = (Input[0] + Input[1] + ... + Input[i]) mod 10

**Alphabet:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (decimal digits)

**Modulus:** 10

**Input example (N=8, k=4):**
```
2314|5298
```

**Output example:**
```
2560|5764
```
(Calculation:
- Pos 0: sum=2 → 2 mod 10 = 2
- Pos 1: sum=2+3=5 → 5 mod 10 = 5
- Pos 2: sum=5+1=6 → 6 mod 10 = 6
- Pos 3: sum=6+4=10 → 10 mod 10 = 0
- Pos 4: sum=10+5=15 → 15 mod 10 = 5
- Pos 5: sum=15+2=17 → 17 mod 10 = 7
- Pos 6: sum=17+9=26 → 26 mod 10 = 6
- Pos 7: sum=26+8=34 → 34 mod 10 = 4
)

**Prompt (complete):**
```
You are a deterministic sequence transformer.
Task: Running sum modulo 10.
Alphabet: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (decimal digits)
Modulus: 10
Rule: At each position, output the cumulative sum of all digits so far, modulo 10.
Sum all digits from the start up to this position, divide by 10, output the remainder.
Example: Input 2314 → Output 2560 (cumulative sums: 2, 2+3=5, 5+1=6, 6+4=10→0)
Constraint: The input contains vertical bars (|). These are structural boundaries. 
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 2314|5298
Output: 
```

---

### Task 3.2: Integer Addition ✓ (Already have)

**Rule:** Add two N-digit numbers digit-by-digit (standard addition)

**Alphabet:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (decimal digits)

**Input format:** digit₁ digit₂ ... digitₙ + digit₁ digit₂ ... digitₙ

**Input example (N=4, k=2):**
```
12|34+56|78
```

**Output example:**
```
6912
```
(Calculation: 1234 + 5678 = 6912. Note: boundaries and operator "+" are ignored when computing.)

**Prompt (complete):**
```
You are a deterministic arithmetic calculator.
Task: Add two 4-digit numbers.
Rule: Treat the two sequences as numbers and output their sum. Add them using standard integer addition (with carry).
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing. The + symbol is the operator and is ignored when summing.
Output ONLY the result of adding the two numbers, zero-padded to 4 digits if needed. No spaces, no explanations, no intermediate work.

Input: 12|34+56|78
Output: 
```

---

## Summary Table

| Tier | Task | Dependency Window | Direction Alignment | Error-Recovery | Status | Alphabet |
|------|------|------------------|-------------------|-----------------|--------|----------|
| **1** | Cyclic Shift | Position-local | N/A | ✓ Yes | ✓ Have | A-Z |
| **1** | Character Substitution | Position-local | N/A | ✓ Yes | NEW | A-Z |
| **2** | Running XOR | Current + Prev | N/A | ✗ No | NEW | Binary (0,1) |
| **2** | Running Sum (Local) | Current + Prev | N/A | ✗ No | NEW | Digits (0-9) |
| **3a** | Running Sum Mod 10 | All Previous | Direction-aligned | ✗ No | ✓ Have | Digits (0-9) |
| **3b** | Integer Addition | All Previous | Direction-misaligned | ✗ No | ✓ Have | Digits (0-9) |

---

## Tier 3: Critical Distinction

Both Tier 3 tasks are **global-state** (depend on all previous positions), but differ on **dependency direction**:

- **Task 3.1 (Running Sum Mod 10):** Direction-aligned. The cumulative dependency flows left-to-right, matching boundary intervals naturally. However, **no error-recovery benefit** because a single wrong cumulative sum corrupts all downstream states—boundaries cannot undo the mistake.

- **Task 3.2 (Integer Addition):** Direction-misaligned. The carry dependency flows right-to-left, directly opposing boundary intervals. **No error-recovery benefit**, compounded by the fact that boundaries actively interfere with carry propagation.

**Key insight:** Error recovery (✓ Yes in Tier 1) is a property of position-local tasks only. Global-state tasks (Tier 3) show ✗ No recovery regardless of direction because a single error corrupts all downstream cumulative state. The direction distinction (3a vs 3b) explains *severity* of accuracy loss, not presence/absence of recovery.

---

## Grid Parameters (All Tasks)

- **Sequence lengths:** N ∈ {4, 8, 16, 32, 64}
- **Boundary intervals:** k ∈ {0, 1, 2, 3, 4, 8, 16}
- **Trials per config:** 10
- **Models:** Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct
- **Prompting:** Chat-templated (using `tokenizer.apply_chat_template`)

