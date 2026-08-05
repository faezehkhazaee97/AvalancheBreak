# Appendix content for Fable — Appendices A, E, F, H

One required correction before anything else: change **"45 task×model"** -> **"43 task×model"**
and **"45 of 45"** -> **"43 of 43"** in the abstract, Section 3.2, and the Discussion. Nothing
else in those sentences changes -- 0.76, 0.50, -154.7, 0.48, 0.56 are all correct as written.
(Root cause: the original 45-row table included 2 rows of Qwen2.5-72B-Instruct, a model never
documented anywhere else in the paper; removing them drops the count to 43 without moving any
other cited number.)

---

## Appendix A: Prompt templates

8 remaining task templates (cyclic_shift already appears in the main text) + all 5 reset-variant
original-vs-reset pairs, shown on identical inputs (verified programmatically, not just visually).
```
======================================================================
TASK: character_substitution  (N=8, k=2, axis=4)
======================================================================
You are a deterministic sequence transformer.
Task: Character substitution mapping.
Alphabet: A, B, C, D
Mapping: A→C, B→D, C→A, D→B
Rule: Replace each character with its mapped character. This is a fixed one-to-one substitution cipher.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: CA|BA|CC|DA
Output: 

======================================================================
TASK: random_permutation  (N=8, k=2, axis=4)
======================================================================
You are a deterministic sequence transformer.
Task: Character substitution mapping.
Alphabet: A, B, C, D
Mapping: A→D, B→C, C→A, D→B
Rule: Replace each character with its mapped character. This is a fixed one-to-one substitution cipher.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: CA|BA|CC|DA
Output: 

======================================================================
TASK: conditional_shift  (N=8, k=2, axis=4)
======================================================================
You are a deterministic sequence transformer.
Task: Branching character shift.
Alphabet: A, B, C, D (positions numbered 0, 1, 2, ... in this order)
Rule: For each character, look at its position number in the alphabet.
- If the position is EVEN, replace the character with the NEXT character in the alphabet (e.g. A -> B).
- If the position is ODD, replace the character with the PREVIOUS character in the alphabet (e.g. B -> A).
Wrap around the alphabet if needed.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: CA|BA|CC|DA
Output: 

======================================================================
TASK: composed_permutation  (N=8, k=2, axis=4)
======================================================================
You are a deterministic sequence transformer.
Task: Composed character substitution (apply two fixed mappings in sequence).
Alphabet: A, B, C, D
Step 1 mapping: A→B, B→A, C→D, D→C
Step 2 mapping: A→D, B→C, C→A, D→B
Rule: For each character, first apply the Step 1 mapping, then apply the Step 2 mapping to that result.
This gives the final output character. Apply this to every character independently.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: CA|BA|CC|DA
Output: 

======================================================================
TASK: running_xor  (N=8, k=2)
======================================================================
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

Input: 00|10|00|00
Output: 

======================================================================
TASK: running_sum_local  (N=8, k=2, axis=4)
======================================================================
You are a deterministic sequence transformer.
Task: Running sum with previous input (local window).
Alphabet: 0-3 (digits)
Modulus: 4
Rule: At each position, output the sum of the current input digit and the previous input digit, modulo 4.
The first digit is copied as-is (no previous input to add).
Each subsequent digit: output[i] = (input[i] + input[i-1]) mod 4
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 00|21|11|00
Output: 

======================================================================
TASK: copy_previous  (N=8, k=2)
======================================================================
You are a deterministic sequence transformer.
Task: Copy the previous input digit.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the PREVIOUS input digit (not the current one).
The first digit is copied as-is (no previous input to copy).
Each subsequent digit: output[i] = input[i-1]
Example: Input 1011 -> Output 1101 (calculation: 1 (copied), then 1, 0, 1 -- each shifted from the position before)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 00|10|00|00
Output: 

======================================================================
TASK: running_max  (N=8, k=2, axis=4)
======================================================================
You are a deterministic sequence transformer.
Task: Running maximum.
Alphabet: 0-3
Rule: At each position, output the LARGEST digit seen so far, including the current position.
Scan left to right; once a digit is output, it never decreases.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 00|21|11|00
Output: 

======================================================================
running_xor vs running_xor_reset  (N=8, k=2)  same-input check: True
======================================================================
--- ORIGINAL ---
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

Input: 00|10|00|00
Output: 
--- RESET ---
You are a deterministic sequence transformer.
Task: Running XOR with previous input, reset at boundaries.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the previous input digit.
The first digit is copied as-is (no previous input to XOR with).
Each subsequent digit: output[i] = input[i] XOR input[i-1]
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately after each boundary, treat that position exactly like the first digit of the whole input -- copy it as-is (nothing before a boundary counts as "previous"), then resume the XOR rule from there until the next boundary.
Example: Input 1011|0010 → Output 1110|0011 (first block: 1 copied, 0⊕1=1, 1⊕0=1, 1⊕1=0; second block restarts: 0 copied, 0⊕0=0, 1⊕0=1, 0⊕1=1)
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: 00|10|00|00
Output: 

======================================================================
copy_previous vs copy_previous_reset  (N=8, k=2)  same-input check: True
======================================================================
--- ORIGINAL ---
You are a deterministic sequence transformer.
Task: Copy the previous input digit.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the PREVIOUS input digit (not the current one).
The first digit is copied as-is (no previous input to copy).
Each subsequent digit: output[i] = input[i-1]
Example: Input 1011 -> Output 1101 (calculation: 1 (copied), then 1, 0, 1 -- each shifted from the position before)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 00|10|00|00
Output: 
--- RESET ---
You are a deterministic sequence transformer.
Task: Copy the previous input digit, reset at boundaries.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the PREVIOUS input digit (not the current one).
The first digit is copied as-is (no previous input to copy).
Each subsequent digit: output[i] = input[i-1]
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately after each boundary, treat that position exactly like the first digit of the whole input -- copy IT as-is (nothing before a boundary counts as "previous"), then resume the shift-by-one rule from there until the next boundary.
Example: Input 1011|0010 → Output 1101|0001 (first block: 1 copied, then 1,0,1 shifted from the position before; second block restarts: 0 copied, then 0,0,1 shifted from the position before, within that block only)
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: 00|10|00|00
Output: 

======================================================================
running_sum_local vs running_sum_local_reset  (N=8, k=2)  same-input check: True
======================================================================
--- ORIGINAL ---
You are a deterministic sequence transformer.
Task: Running sum with previous input (local window).
Alphabet: 0-3 (digits)
Modulus: 4
Rule: At each position, output the sum of the current input digit and the previous input digit, modulo 4.
The first digit is copied as-is (no previous input to add).
Each subsequent digit: output[i] = (input[i] + input[i-1]) mod 4
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 00|21|11|00
Output: 
--- RESET ---
You are a deterministic sequence transformer.
Task: Running sum with previous input (local window), reset at boundaries.
Alphabet: 0-3 (digits)
Modulus: 4
Rule: At each position, output the sum of the current input digit and the previous input digit, modulo 4.
The first digit is copied as-is (no previous input to add).
Each subsequent digit: output[i] = (input[i] + input[i-1]) mod 4
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately after each boundary, treat that position exactly like the first digit of the whole input -- copy it as-is (nothing before a boundary counts as "previous"), then resume the running-sum rule from there until the next boundary.
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: 00|21|11|00
Output: 

======================================================================
reverse_xor vs reverse_xor_reset  (N=8, k=2)  same-input check: True
======================================================================
--- ORIGINAL ---
You are a deterministic sequence transformer.
Task: Running XOR with next input.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the NEXT input digit.
The last digit is copied as-is (no next input to XOR with).
Each other digit: output[i] = input[i] XOR input[i+1]
Example: Input 1011 → Output 1101 (calculation: 1⊕0=1, 0⊕1=1, 1⊕1=0, 1)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 00|10|00|00
Output: 
--- RESET ---
You are a deterministic sequence transformer.
Task: Running XOR with next input, reset at boundaries.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the NEXT input digit.
The last digit is copied as-is (no next input to XOR with).
Each other digit: output[i] = input[i] XOR input[i+1]
IMPORTANT: The input contains vertical bars (|), structural boundaries. Immediately BEFORE each boundary, treat that position exactly like the last digit of the whole input -- copy it as-is (nothing after a boundary counts as "next" for it), then the rule resumes normally on the other side of the boundary.
Example: Input 1011|0010 → Output 1101|0110 (first block: 1⊕0=1, 0⊕1=1, 1⊕1=0, 1 copied since it's right before the boundary; second block: 0⊕0=0, 0⊕1=1, 1⊕0=1, 0 copied since it's the last digit)
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: 00|10|00|00
Output: 

======================================================================
running_max vs running_max_reset  (N=8, k=2)  same-input check: True
======================================================================
--- ORIGINAL ---
You are a deterministic sequence transformer.
Task: Running maximum.
Alphabet: 0-3
Rule: At each position, output the LARGEST digit seen so far, including the current position.
Scan left to right; once a digit is output, it never decreases.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: 00|21|11|00
Output: 
--- RESET ---
You are a deterministic sequence transformer.
Task: Running maximum, reset at boundaries.
Alphabet: 0-3
Rule: At each position, output the LARGEST digit seen so far, including the current position.
Scan left to right; once a digit is output, it never decreases -- within the current block.
IMPORTANT: The input contains vertical bars (|), structural boundaries. Each boundary starts a brand new block: the running maximum RESETS to nothing at that point and starts counting fresh from the first digit of the new block, ignoring everything before the boundary.
Copy the boundary characters themselves exactly to the output.
No spaces, no explanations.

Input: 00|21|11|00
Output: 

```

---

## Appendix E: Dose-response fit comparison

### 45-row aggregate table, CORRECTED to 43 rows (Qwen2.5-72B-Instruct removed)
```csv
Task,Model,N_combos,lin_r2,quad_r2,exp_r2
cyclic_shift,Mistral-7B-Instruct-v0.3,18,0.6469912464227473,0.7481863683843023,-4.531388065067961
cyclic_shift,Phi-3.5-mini-instruct,18,0.5221658337302566,0.7374952840396669,-7.08370039248504
cyclic_shift,Qwen2.5-14B-Instruct,18,0.6009218704913025,0.9085362683891122,-2.169400643097828
cyclic_shift,Qwen2.5-32B-Instruct,18,0.5541321182477497,0.9176100275900448,-3.197628708005911
cyclic_shift,Qwen2.5-7B-Instruct,18,0.5760099916434941,0.8033598261629626,-3.104924334917349
character_substitution,Mistral-7B-Instruct-v0.3,12,0.772232640166972,0.915658573201898,-4.665214857653918
character_substitution,Phi-3.5-mini-instruct,12,0.5339341996218305,0.7219181403222841,-9.216205729654948
character_substitution,Qwen2.5-14B-Instruct,12,0.8495456323758669,0.9765987467641016,-0.7879873704226784
character_substitution,Qwen2.5-32B-Instruct,12,0.8496149354488276,0.9810668524374992,-1.155272074268359
character_substitution,Qwen2.5-7B-Instruct,12,0.7885196333973465,0.9363060086546184,-1.4737944255865731
random_permutation,Mistral-7B-Instruct-v0.3,12,0.7541923627107728,0.9321224890116075,-5.371249685956872
random_permutation,Phi-3.5-mini-instruct,12,0.60331103250454,0.7395807212793192,-12.685477114335868
random_permutation,Qwen2.5-14B-Instruct,12,0.8497448561534003,0.981569763217916,-0.8728079900563124
random_permutation,Qwen2.5-32B-Instruct,12,0.8487458541101064,0.9825936773335182,-1.2461671617573076
random_permutation,Qwen2.5-7B-Instruct,12,0.8583690685489902,0.9568718400786668,-1.3883925398155892
conditional_shift,Mistral-7B-Instruct-v0.3,12,0.7004750775302203,0.8949231924051951,-3.217476217033178
conditional_shift,Phi-3.5-mini-instruct,12,0.6161219560739236,0.8385818894547293,-3.829393420781555
conditional_shift,Qwen2.5-14B-Instruct,12,0.8932733860094891,0.9377451795031034,-0.5233381630974493
conditional_shift,Qwen2.5-32B-Instruct,12,0.8187373035742995,0.948151796175337,-3.0977562314089213
conditional_shift,Qwen2.5-7B-Instruct,12,0.5723321928026671,0.7363404598398668,-7.428447587509789
composed_permutation,Mistral-7B-Instruct-v0.3,8,0.376935695105244,0.8818511286462174,-12.921288840009776
composed_permutation,Phi-3.5-mini-instruct,8,0.3492016510922013,0.5565566987799638,-12.754482377190795
composed_permutation,Qwen2.5-14B-Instruct,8,0.5884142703509878,0.8179788077025282,-31.201785492182548
composed_permutation,Qwen2.5-32B-Instruct,8,0.6911097051584161,0.8547946971434702,-3.6557816073588025
composed_permutation,Qwen2.5-7B-Instruct,8,0.7244007719311684,0.8784448962545004,-5.13099151536243
running_xor,Mistral-7B-Instruct-v0.3,4,0.2491267849930729,0.6563750526696184,-20.077698482431853
running_xor,Phi-3.5-mini-instruct,4,0.2008355577292503,0.382737236198518,-95.28307489289064
running_xor,Qwen2.5-14B-Instruct,4,0.104745823354114,0.3241689303651809,-72.53975206526528
running_xor,Qwen2.5-32B-Instruct,4,0.296643864467902,0.4526587056477864,-59.010066022151904
running_xor,Qwen2.5-7B-Instruct,4,0.4844273774293062,0.5804387465301448,-10.850729061212313
running_sum_local,Mistral-7B-Instruct-v0.3,12,0.2258137113920036,0.5375394689061687,-29.7058378727436
running_sum_local,Phi-3.5-mini-instruct,12,0.5317291292964246,0.8667138620444678,-17.533133083460992
running_sum_local,Qwen2.5-14B-Instruct,12,0.2029680925581386,0.3165364519920009,-26.112763020940847
running_sum_local,Qwen2.5-32B-Instruct,12,0.3961630978006047,0.6273830142852023,-26.19213944395513
running_sum_local,Qwen2.5-7B-Instruct,12,0.1578500984099667,0.4530082267895571,-22.22766901919576
copy_previous,Mistral-7B-Instruct-v0.3,4,0.2040793554749757,0.6011958586651527,-154.65516133221664
copy_previous,Qwen2.5-14B-Instruct,4,0.1189932260470997,0.8870027609782899,-35.58173394682773
copy_previous,Qwen2.5-32B-Instruct,4,0.0803706527110091,0.7281343827005513,-33.80105533266618
copy_previous,Qwen2.5-7B-Instruct,4,0.5757017704675585,0.8363629074036536,-13.817731983355763
running_max,Mistral-7B-Instruct-v0.3,12,0.1131409362050829,0.6691771022734762,-4.179742072099786
running_max,Qwen2.5-14B-Instruct,12,0.1716796054508749,0.841745353707764,-38.8456568248746
running_max,Qwen2.5-32B-Instruct,12,0.1445240516554622,0.5670428134985496,-73.01157478758833
running_max,Qwen2.5-7B-Instruct,12,0.4353990475876033,0.753185106313755,-20.877955568272785
```

### 348-curve per-curve robustness table (AIC/BIC/CV winners, vertex + CI)
```csv
Task,Model,Seq_Length,Axis_Val,N_Densities,linear_AIC,linear_BIC,quadratic_AIC,quadratic_BIC,exponential_AIC,exponential_BIC,AIC_Winner,BIC_Winner,CV_MSE_Quadratic,CV_MSE_Linear,CV_Favors_Quadratic,Vertex,Vertex_CI_Low,Vertex_CI_High
cyclic_shift,Mistral-7B-Instruct-v0.3,16,2.0,5,-26.597821446593322,-27.378945621725123,-37.78683308920775,-38.958519351905444,-9.811448760976901,-10.5925729361087,quadratic,quadratic,,,False,0.23576388888888894,0.13073470744680846,0.47249999999999953
cyclic_shift,Mistral-7B-Instruct-v0.3,16,4.0,5,-32.8275500720629,-33.6086742471947,-42.24737297810945,-43.41905924080715,-14.754842599947171,-15.53596677507897,quadratic,quadratic,,,False,-0.12609060402684572,-0.47495303055056365,0.11250000000000003
cyclic_shift,Mistral-7B-Instruct-v0.3,16,8.0,5,-28.745523743686334,-29.526647918818135,-29.068269484486116,-30.239955747183814,-15.520548422999752,-16.301672598131553,quadratic,quadratic,,,False,-0.47225186766275323,-0.47225186766275334,1.0181962025316458
cyclic_shift,Mistral-7B-Instruct-v0.3,32,2.0,6,-37.063069104246026,-37.47955016578992,-39.80779612203572,-40.432517714351555,-15.487209626635078,-15.903690688178969,quadratic,quadratic,0.001565076410066882,0.004739727492982158,True,0.03797703437560672,-0.45666716893959236,0.5686500369094457
cyclic_shift,Mistral-7B-Instruct-v0.3,32,4.0,6,-36.37326644634021,-36.7897475078841,-64.98838805981894,-65.61310965213478,-21.138419161088294,-21.554900222632185,quadratic,quadratic,1.9483751478977278e-05,0.007775402474840584,True,0.2571111312545321,0.11103723404255363,0.32772866326648753
cyclic_shift,Mistral-7B-Instruct-v0.3,32,8.0,6,-34.837346339897856,-35.25382740144175,-34.50572783920359,-35.13044943151942,-22.848784765149265,-23.265265826693156,linear,linear,0.08050981700099065,0.005840122746750193,False,-0.22966442749939583,-0.3367200441056157,1.3175467100298124
cyclic_shift,Mistral-7B-Instruct-v0.3,64,2.0,6,-34.32266228936288,-34.73914335090677,-33.783953785369036,-34.40867537768487,-17.678309575547047,-18.094790637090938,linear,linear,0.029474942658807967,0.003804837740194554,False,0.20233318997296612,-0.4164600052624619,0.8093578023298041
cyclic_shift,Mistral-7B-Instruct-v0.3,64,4.0,6,-42.37819265953182,-42.79467372107571,-49.72297400399859,-50.34769559631442,-23.46849374372169,-23.884974805265582,quadratic,quadratic,0.007752777030863298,0.002073123579997248,False,0.1947251698682995,-0.05084792758154501,0.343532028734947
cyclic_shift,Mistral-7B-Instruct-v0.3,64,8.0,6,-46.780211514943815,-47.196692576487706,-44.85619672781793,-45.48091832013377,-28.397778708014158,-28.81425976955805,linear,linear,0.012321509258261462,0.00041796538449830273,False,-3.5925635186961955,,
cyclic_shift,Mistral-7B-Instruct-v0.3,128,2.0,6,-33.20048858527448,-33.61696964681837,-31.410917728789272,-32.03563932110511,-15.737185405014039,-16.15366646655793,linear,linear,0.1712611510306217,0.0036586554762610304,False,0.39229260089686085,-0.09947373283679842,0.8580904292205783
cyclic_shift,Mistral-7B-Instruct-v0.3,128,4.0,6,-36.701932175502655,-37.118413237046546,-35.48390005467845,-36.10862164699429,-24.84821766638801,-25.2646987279319,linear,linear,0.06090551497086499,0.0019400737072991791,False,0.3429254956989597,-0.17326226013874535,1.1922021943573664
cyclic_shift,Mistral-7B-Instruct-v0.3,128,8.0,6,-51.01023349170972,-51.42671455325361,-54.70461561177173,-55.329337204087565,-31.780377511427517,-32.19685857297141,quadratic,quadratic,0.0035002051814649548,0.0006054045810000558,False,0.11156735561418768,-0.005514705882353232,0.4273255813953488
cyclic_shift,Mistral-7B-Instruct-v0.3,256,2.0,6,-36.716039659673605,-37.132520721217496,-35.53420638796368,-36.158927980279515,-17.56239711627388,-17.97887817781777,linear,linear,0.07389591142683387,0.0036281935209810275,False,0.6406795818208455,0.1290394264637474,0.9821035242290752
cyclic_shift,Mistral-7B-Instruct-v0.3,256,4.0,6,-38.01099461783119,-38.42747567937508,-37.82725114142136,-38.45197273373719,-25.246388452680662,-25.662869514224553,linear,quadratic,0.04423036753678731,0.0019736185273554383,False,0.42011050974778613,-0.09063416422287342,0.7984813084112151
cyclic_shift,Mistral-7B-Instruct-v0.3,256,8.0,6,-46.10526057626442,-46.52174163780831,-44.32338064116679,-44.94810223348263,-34.83783866336006,-35.25431972490395,linear,linear,0.0031634825661620357,0.00039779995332051883,False,-0.07083095109974655,-0.27229568630326817,1.2142203660821556
cyclic_shift,Phi-3.5-mini-instruct,16,2.0,5,-40.610850687224485,-41.391974862356285,-39.728276280237765,-40.89996254293546,-13.22143376712124,-14.00255794225304,linear,linear,,,False,2.8304761904761837,,
cyclic_shift,Phi-3.5-mini-instruct,16,4.0,5,-36.24404151632921,-37.02516569146101,-49.0095300071579,-50.1812162698556,-16.578203916783412,-17.359328091915213,quadratic,quadratic,,,False,0.043582036775106046,-0.1944444444444458,0.812499999999996
cyclic_shift,Phi-3.5-mini-instruct,16,8.0,5,-27.669911826179742,-28.451036001311543,-26.08836667327172,-27.26005293596942,-15.361105575735355,-16.142229750867156,linear,linear,,,False,-1.164407216494843,,
cyclic_shift,Phi-3.5-mini-instruct,32,2.0,6,-34.53008214176755,-34.94656320331144,-36.34566843677486,-36.970390029090694,-16.72386475024935,-17.140345811793242,quadratic,quadratic,0.001908735075936954,0.006308551623300142,True,0.372992458479268,0.03964552238805965,0.5873443060498231
cyclic_shift,Phi-3.5-mini-instruct,32,4.0,6,-37.38852524474813,-37.80500630629202,-42.29020598533339,-42.91492757764922,-23.482674894373055,-23.899155955916946,quadratic,quadratic,0.016080492422951575,0.00586602931383278,False,0.322961277658815,-0.126312895902015,0.6094635627530361
cyclic_shift,Phi-3.5-mini-instruct,32,8.0,6,-37.587830916920794,-38.004311978464685,-44.887855405284164,-45.5125769976,-25.409932194643485,-25.826413256187376,quadratic,quadratic,0.0005506278046428073,0.005235039433342814,True,0.781075371789094,-0.14388954383601077,0.9034435673365427
cyclic_shift,Phi-3.5-mini-instruct,64,2.0,6,-35.43283705073928,-35.84931811228317,-35.559166612883985,-36.18388820519982,-17.718888521431637,-18.135369582975528,quadratic,quadratic,0.008039610412433952,0.0040742372105521925,False,0.44137332695984693,0.1441532258064513,1.0917798250733595
cyclic_shift,Phi-3.5-mini-instruct,64,4.0,6,-48.142135875315105,-48.558616936858996,-48.642001994490016,-49.26672358680585,-25.24173203647961,-25.658213098023502,quadratic,quadratic,0.005693503140015367,0.0007136899169922288,False,0.29924535603715124,-0.20389024166129907,0.6712499999999995
cyclic_shift,Phi-3.5-mini-instruct,64,8.0,6,-35.87305284701893,-36.289533908562824,-34.419222280303586,-35.04394387261942,-29.46507210522701,-29.8815531667709,linear,linear,0.09300332508483355,0.0020832588780342787,False,0.8537191448721709,-0.3947368421052631,1.2914557781335412
cyclic_shift,Phi-3.5-mini-instruct,128,2.0,6,-33.599786513529736,-34.01626757507363,-38.59097633133213,-39.215697923647966,-23.018734951028605,-23.435216012572496,quadratic,quadratic,0.0034294072801778785,0.009477449512408196,True,0.5940228844699017,0.12769555591997386,0.8095153082180975
cyclic_shift,Phi-3.5-mini-instruct,128,4.0,6,-44.58496593883552,-45.001447000379414,-46.55085088822278,-47.17557248053862,-26.158948332283394,-26.575429393827285,quadratic,quadratic,0.011851973215421238,0.0015653654598069909,False,0.44317040185471474,-0.08663549156403885,0.6208397806610639
cyclic_shift,Phi-3.5-mini-instruct,128,8.0,6,-47.13600338253882,-47.55248444408271,-45.85113364359514,-46.47585523591098,-33.7320194241405,-34.148500485684394,linear,linear,0.008645931837877648,0.00040867615170095184,False,0.8717990827962304,0.10425064399179339,1.3124843043696726
cyclic_shift,Phi-3.5-mini-instruct,256,2.0,6,-32.167634977242244,-32.584116038786135,-32.31544401628625,-32.940165608602086,-19.974675397193977,-20.391156458737868,quadratic,quadratic,0.12435434815746542,0.010242918106852378,False,0.6427350311126582,-0.18154069767441902,0.7340671862329394
cyclic_shift,Phi-3.5-mini-instruct,256,4.0,6,-50.03449857341511,-50.450979634959,-54.15389862991495,-54.778620222230785,-28.277193898919847,-28.69367496046374,quadratic,quadratic,0.0023307850662504753,0.0006884711888055587,False,0.6374851151098172,-0.09375986075144781,0.7752785754712801
cyclic_shift,Phi-3.5-mini-instruct,256,8.0,6,-48.29684027406721,-48.7133213356111,-53.61200368717867,-54.23672527949451,-35.3250333039525,-35.74151436549639,quadratic,quadratic,0.0010702826525340197,0.0007783883001763071,False,0.5144097600872413,0.1666158536585366,0.5805545649072752
cyclic_shift,Qwen2.5-14B-Instruct,16,2.0,5,-21.644191699097597,-22.425315874229398,-26.75657756564923,-27.928263828346928,-8.290685184473784,-9.071809359605583,quadratic,quadratic,,,False,0.7911710719530102,-0.1147151898734188,1.3173076923076894
cyclic_shift,Qwen2.5-14B-Instruct,16,4.0,5,-34.287777432028335,-35.068901607160136,-34.741890893006115,-35.91357715570381,-11.722127571722178,-12.503251746853977,quadratic,quadratic,,,False,2.617636363636363,,
cyclic_shift,Qwen2.5-14B-Instruct,16,8.0,5,-29.601396927024993,-30.382521102156794,-30.664393291680476,-31.836079554378173,-9.463981351486108,-10.245105526617907,quadratic,quadratic,,,False,1.4321335078534048,-0.03383027522935802,1.450000000000002
cyclic_shift,Qwen2.5-14B-Instruct,32,2.0,6,-28.05012134662141,-28.4666024081653,-38.59898671044223,-39.22370830275807,-12.8684245972416,-13.28490565878549,quadratic,quadratic,0.029388998188217153,0.03088160987829493,True,0.8376124632157848,-0.363013820644951,0.9789694908313239
cyclic_shift,Qwen2.5-14B-Instruct,32,4.0,6,-34.1210450084905,-34.53752607003439,-42.49108200319539,-43.115803595511224,-18.29443397540833,-18.710915036952223,quadratic,quadratic,0.012840278687957974,0.010843801521564467,False,1.310647249525068,-0.3968750000000003,1.4649748322147655
cyclic_shift,Qwen2.5-14B-Instruct,32,8.0,6,-23.878688006267666,-24.295169067811557,-33.97906154667547,-34.603783138991304,-17.624079838373245,-18.040560899917136,quadratic,quadratic,0.004554188410217177,0.05662964788001521,True,0.7606585146479355,-0.025244104642594176,1.0355290456431532
cyclic_shift,Qwen2.5-14B-Instruct,64,2.0,6,-22.738275620624385,-23.154756682168276,-32.807933950247346,-33.43265554256318,-13.492006035718589,-13.90848709726248,quadratic,quadratic,0.12140263679404245,0.07649643746115159,False,0.6699188797456519,-0.20933406113537067,1.081798245614035
cyclic_shift,Qwen2.5-14B-Instruct,64,4.0,6,-25.407016885618347,-25.823497947162238,-42.26098719747913,-42.88570878979497,-21.39402889307094,-21.81050995461483,quadratic,quadratic,0.009651307153892409,0.049027183439363554,True,0.7383150965937507,0.23489010989010992,1.184707102381488
cyclic_shift,Qwen2.5-14B-Instruct,64,8.0,6,-19.932187869798284,-20.348668931342175,-31.588130399041304,-32.21285199135714,-21.624805535027484,-22.041286596571375,quadratic,quadratic,0.04347934328145373,0.1054770875476345,True,0.6242184939954313,0.02594557195571955,0.6524061082171063
cyclic_shift,Qwen2.5-14B-Instruct,128,2.0,6,-25.798363076641117,-26.214844138185008,-42.80333660138334,-43.42805819369918,-14.981078755942793,-15.397559817486684,quadratic,quadratic,0.01077107861819767,0.04612151923637703,True,0.5766966777610743,0.21448442136498522,1.3793263739401596
cyclic_shift,Qwen2.5-14B-Instruct,128,4.0,6,-26.980419176836467,-27.396900238380358,-43.557067240530785,-44.18178883284662,-26.79379431508282,-27.21027537662671,quadratic,quadratic,0.016734479153328517,0.03416727565764215,True,0.6214897259594655,0.27883270466589133,0.6355395551933493
cyclic_shift,Qwen2.5-14B-Instruct,128,8.0,6,-24.48019496466025,-24.89667602620414,-29.215638748341547,-29.840360340657384,-25.641720199434197,-26.058201260978088,quadratic,quadratic,0.20794294944097372,0.0355679740737429,False,0.5757045552004738,0.20244565217391305,0.6398131716591455
cyclic_shift,Qwen2.5-14B-Instruct,256,2.0,6,-33.942697856220455,-34.359178917764346,-38.56560205032418,-39.19032364264002,-15.032216279345093,-15.448697340888984,quadratic,quadratic,0.028011403694223985,0.010291045373612582,False,0.4894306282722513,-0.12428573873510428,1.0033312182741136
cyclic_shift,Qwen2.5-14B-Instruct,256,4.0,6,-32.55893686621164,-32.97541792775553,-51.152805142399004,-51.77752673471484,-28.709175909391575,-29.125656970935466,quadratic,quadratic,0.004688610597774271,0.01367309395931196,True,0.6120381532775824,0.3045058139534885,0.6655694365164315
cyclic_shift,Qwen2.5-14B-Instruct,256,8.0,6,-29.054900165344392,-29.471381226888283,-31.404179314636224,-32.02890090695206,-29.38386363226821,-29.8003446938121,quadratic,quadratic,0.1601042656413431,0.013620702167929169,False,0.49177586081482577,0.19612562068631129,0.5863914373088686
cyclic_shift,Qwen2.5-32B-Instruct,16,2.0,5,-24.09627323937939,-24.87739741451119,-30.059684859629925,-31.231371122327623,-7.284975647804952,-8.066099822936751,quadratic,quadratic,,,False,0.7877976190476191,-0.44969512195121486,1.261363636363635
cyclic_shift,Qwen2.5-32B-Instruct,16,4.0,5,-22.89497927857442,-23.676103453706222,-44.50743285983099,-45.67911912252869,-10.866336577363192,-11.647460752494991,quadratic,quadratic,,,False,0.8745950127411721,0.8437499999999999,1.3124999999999987
cyclic_shift,Qwen2.5-32B-Instruct,16,8.0,5,-24.251092833522282,-25.032217008654083,-25.666264215305336,-26.837950478003037,-8.637378247846849,-9.418502422978648,quadratic,quadratic,,,False,0.9636504424778762,-0.04802631578947203,1.3106673902213373
cyclic_shift,Qwen2.5-32B-Instruct,32,2.0,6,-27.952886895050945,-28.369367956594836,-35.24054228005593,-35.86526387237176,-12.184432684081724,-12.600913745625615,quadratic,quadratic,0.09137080681369748,0.0314085348951771,False,0.828019918206604,-0.381430938922323,1.3694852941176516
cyclic_shift,Qwen2.5-32B-Instruct,32,4.0,6,-30.43228654787363,-30.84876760941752,-50.93249833851755,-51.557219930833384,-16.091324302764036,-16.507805364307927,quadratic,quadratic,0.0019377876904959305,0.020137316537709348,True,0.9728637123842045,0.38437500000000024,1.086044520547944
cyclic_shift,Qwen2.5-32B-Instruct,32,8.0,6,-25.384742832080413,-25.801223893624304,-33.109158409147156,-33.73388000146299,-16.025757044700622,-16.442238106244513,quadratic,quadratic,0.010431453080103001,0.03967264949551775,True,0.8522502422871913,-0.04440789473684226,1.2072538528350065
cyclic_shift,Qwen2.5-32B-Instruct,64,2.0,6,-24.219486538887857,-24.635967600431748,-34.375774972116545,-35.00049656443238,-13.3120672542398,-13.72854831578369,quadratic,quadratic,0.0458639875221713,0.05818128567662193,True,0.676943602930816,-0.1499547160215825,0.8383858012372774
cyclic_shift,Qwen2.5-32B-Instruct,64,4.0,6,-22.75647742889553,-23.17295849043942,-39.94959824528208,-40.57431983759792,-21.12683693371003,-21.543317995253922,quadratic,quadratic,0.03244959239394398,0.06932896753768976,True,0.6716387375153833,0.3028234728318847,0.688741622786022
cyclic_shift,Qwen2.5-32B-Instruct,64,8.0,6,-17.902185724057198,-18.31866678560109,-32.40493743292648,-33.02965902524232,-21.11095268833228,-21.52743374987617,quadratic,quadratic,0.10512755730055336,0.14959485884834703,True,0.6268295937547755,0.2756529850746271,0.7204056291390728
cyclic_shift,Qwen2.5-32B-Instruct,128,2.0,6,-22.885713339757295,-23.302194401301186,-32.184038351358794,-32.80875994367463,-14.257013125911985,-14.673494187455876,quadratic,quadratic,0.003861057228788288,0.06608917817532163,True,0.5320845560866337,0.17389021508119631,0.9353954227408179
cyclic_shift,Qwen2.5-32B-Instruct,128,4.0,6,-21.911763123424816,-22.328244184968707,-38.557272145211854,-39.18199373752769,-23.870488011630727,-24.286969073174618,quadratic,quadratic,0.037820186972826256,0.07962904703006833,True,0.5887720955441808,0.26906919344845687,0.6018653100831705
cyclic_shift,Qwen2.5-32B-Instruct,128,8.0,6,-18.052355451102315,-18.468836512646206,-25.271479423741738,-25.896201016057574,-22.978353791640046,-23.394834853183937,quadratic,quadratic,0.31143449459449135,0.12367698734729242,False,0.5415939894297729,0.19014444949954504,0.5703561274997111
cyclic_shift,Qwen2.5-32B-Instruct,256,2.0,6,-28.68439176738913,-29.10087282893302,-31.48114917453563,-32.10587076685147,-14.229587171978629,-14.64606823352252,quadratic,quadratic,0.034659001045222015,0.02111400877260754,False,0.4446419044677455,-0.02140374669661687,0.7902114596529174
cyclic_shift,Qwen2.5-32B-Instruct,256,4.0,6,-27.735517965848906,-28.151999027392797,-34.401410419272835,-35.02613201158867,-28.09547219298647,-28.51195325453036,quadratic,quadratic,0.02767378197874711,0.025721109828318017,False,0.5792450077698857,0.1746937161873011,0.610965095089283
cyclic_shift,Qwen2.5-32B-Instruct,256,8.0,6,-21.4963525532848,-21.912833614828692,-22.893698376704876,-23.518419969020712,-24.035509800377092,-24.451990861920983,exponential,exponential,0.5272495909803389,0.04377874004070311,False,0.4929999664331075,0.1755778159063383,1.299106727957554
cyclic_shift,Qwen2.5-7B-Instruct,16,2.0,5,-19.185426143911542,-19.966550319043343,-18.978420188227627,-20.15010645092533,-8.52494331109232,-9.30606748622412,linear,quadratic,,,False,0.9235592257650254,-0.06250000000000011,0.9620493358633778
cyclic_shift,Qwen2.5-7B-Instruct,16,4.0,5,-36.17724228383357,-36.95836645896537,-34.44520500397388,-35.61689126667158,-13.40396550375496,-14.18508967888676,linear,linear,,,False,-6.339296407185603,,
cyclic_shift,Qwen2.5-7B-Instruct,16,8.0,5,-29.500411727156973,-30.281535902288773,-28.39525851065519,-29.566944773352887,-10.691625045856513,-11.472749220988312,linear,linear,,,False,-1.2195623916811102,,
cyclic_shift,Qwen2.5-7B-Instruct,32,2.0,6,-23.763794661356,-24.18027572289989,-24.87533057367949,-25.500052165995328,-12.922936100791961,-13.339417162335852,quadratic,quadratic,0.4073771264441744,0.046347111691159494,False,0.8038575455764742,-0.0001702973957334625,0.8985160378485746
cyclic_shift,Qwen2.5-7B-Instruct,32,4.0,6,-34.40680460023779,-34.82328566178168,-32.88939786610546,-33.5141194584213,-19.96281915231362,-20.379300213857512,linear,linear,0.12747039754951364,0.00266664518221855,False,2.1810016556291383,,
cyclic_shift,Qwen2.5-7B-Instruct,32,8.0,6,-29.598031714185595,-30.014512775729486,-33.11607590599213,-33.74079749830797,-20.004433205571495,-20.420914267115386,quadratic,quadratic,0.05904356188064433,0.014498719909355895,False,0.9271518100448227,-0.023097826086956517,1.135666283166234
cyclic_shift,Qwen2.5-7B-Instruct,64,2.0,6,-22.42161898359992,-22.83810004514381,-23.110411809269515,-23.73513340158535,-13.031016323814786,-13.447497385358677,quadratic,quadratic,0.24026834342675552,0.04986207901432135,False,0.6522619887165022,0.02436367671483929,1.15915586103365
cyclic_shift,Qwen2.5-7B-Instruct,64,4.0,6,-36.923248082259896,-37.33972914380379,-45.42148855762251,-46.04621014993835,-25.449801741491967,-25.86628280303586,quadratic,quadratic,0.012491105935470514,0.005403276487066212,False,0.8871653005464479,0.34975961538461553,1.0474574012457176
cyclic_shift,Qwen2.5-7B-Instruct,64,8.0,6,-22.496310501000288,-22.91279156254418,-26.693062020691002,-27.31778361300684,-25.607637132976897,-26.02411819452079,quadratic,quadratic,0.34026111690349853,0.047529848124239334,False,0.5893297948663443,0.2115536277602524,0.6611162195185113
cyclic_shift,Qwen2.5-7B-Instruct,128,2.0,6,-31.23966740246304,-31.656148464006932,-32.578194799164066,-33.2029163914799,-15.004366558473762,-15.420847620017653,quadratic,quadratic,0.15147723839222796,0.014168225932906542,False,0.579680735930736,0.09971850667951652,0.7182351814247812
cyclic_shift,Qwen2.5-7B-Instruct,128,4.0,6,-41.499508004514176,-41.91598906605807,-39.75922498345469,-40.38394657577053,-27.651198791264466,-28.067679852808357,linear,linear,0.022565687150750025,0.0009033774377861328,False,1.6263219333669878,,
cyclic_shift,Qwen2.5-7B-Instruct,128,8.0,6,-32.22235059460507,-32.63883165614896,-34.59561319048016,-35.220334782795994,-28.270705971156623,-28.687187032700514,quadratic,quadratic,0.10634157306988347,0.007820204929081875,False,0.5297959302713153,0.21843442368710092,0.6421282798833818
cyclic_shift,Qwen2.5-7B-Instruct,256,2.0,6,-34.07603195138799,-34.49251301293188,-55.49883199004299,-56.123553582358824,-17.439213973885785,-17.855695035429676,quadratic,quadratic,0.002001437861845963,0.011692974970897751,True,0.5233232540497754,-0.30089278904819683,0.8076776458842064
cyclic_shift,Qwen2.5-7B-Instruct,256,4.0,6,-43.060712984213595,-43.477194045757486,-56.21068447130875,-56.83540606362459,-29.34062136338769,-29.75710242493158,quadratic,quadratic,0.000265630471802313,0.002321800108052536,True,0.5460242227340022,-0.02254746835442961,0.6290760869565211
cyclic_shift,Qwen2.5-7B-Instruct,256,8.0,6,-38.03782088564202,-38.45430194718591,-39.191436304158415,-39.81615789647425,-32.233256868351575,-32.649737929895466,quadratic,quadratic,0.04012657315407748,0.002603476391453502,False,0.4865316115283137,0.07314850651776038,1.0898981851836367
character_substitution,Mistral-7B-Instruct-v0.3,16,2.0,5,-40.68645603704338,-41.46758021217518,-50.03517405592267,-51.206860318620365,-9.594419969230811,-10.37554414436261,quadratic,quadratic,,,False,-0.8522675560600426,,
character_substitution,Mistral-7B-Instruct-v0.3,16,4.0,5,-29.8588968343293,-30.6400210094611,-45.06956959830633,-46.24125586100403,-14.330174173926455,-15.111298349058254,quadratic,quadratic,,,False,0.0921101256467108,-0.10000000000000032,0.21172480620155013
character_substitution,Mistral-7B-Instruct-v0.3,16,8.0,5,-27.451906843375113,-28.233031018506914,-41.67792293766359,-42.849609200361286,-18.259356043538205,-19.040480218670005,quadratic,quadratic,,,False,-0.05788629737609313,-0.3981613454960113,0.238970588235294
character_substitution,Mistral-7B-Instruct-v0.3,32,2.0,6,-33.00588881474222,-33.42236987628611,-37.174566035370695,-37.79928762768653,-14.574597289625764,-14.991078351169655,quadratic,quadratic,0.06489133093934718,0.012410792856970554,False,0.29327199463441656,-0.11582829281549888,0.519904458598726
character_substitution,Mistral-7B-Instruct-v0.3,32,4.0,6,-43.10034966456788,-43.51683072611177,-52.2591205680403,-52.88384216035614,-21.19307970985823,-21.609560771402123,quadratic,quadratic,0.0006005187669517239,0.002155170460284334,True,0.20948132445327486,0.0050761738056586524,0.44718886462882107
character_substitution,Mistral-7B-Instruct-v0.3,32,8.0,6,-41.148885880653445,-41.565366942197336,-52.28360086904284,-52.90832246135868,-24.533381196198647,-24.949862257742538,quadratic,quadratic,0.0018436157837364063,0.0030677882750544012,True,-0.060119217589949495,-0.3963815789473685,0.21837200917711708
character_substitution,Mistral-7B-Instruct-v0.3,64,2.0,6,-35.8260790184544,-36.24256007999829,-45.63016137786122,-46.25488297017706,-15.176510563296532,-15.592991624840423,quadratic,quadratic,0.007525473262925035,0.006930701135958548,False,0.3944103889608021,-0.06656137184115528,0.4570175204309088
character_substitution,Mistral-7B-Instruct-v0.3,64,4.0,6,-39.57714803673251,-39.9936290982764,-38.22316690115042,-38.84788849346626,-23.707065558075435,-24.123546619619326,linear,linear,0.03986724471436385,0.0012420244315920327,False,0.2569472663465082,-0.21026248418161558,1.098464852277038
character_substitution,Mistral-7B-Instruct-v0.3,64,8.0,6,-40.06259502429622,-40.47907608584011,-50.77767311484061,-51.402394707156446,-27.22194797566084,-27.63842903720473,quadratic,quadratic,0.006277989659229377,0.003404379866501111,False,0.30857640026203664,0.16388349005608205,0.4135416666666667
character_substitution,Phi-3.5-mini-instruct,16,2.0,5,-24.946765643111014,-25.727889818242815,-24.178006783719567,-25.349693046417265,-10.365542708707057,-11.146666883838856,linear,linear,,,False,0.6235714285714286,0.04222972972972943,0.7309633027522929
character_substitution,Phi-3.5-mini-instruct,16,4.0,5,-30.210898490603455,-30.992022665735256,-30.341421741285778,-31.513108003983476,-17.51910885064799,-18.30023302577979,quadratic,quadratic,,,False,0.12589514066496102,-0.3518318860598085,1.0096153846153846
character_substitution,Phi-3.5-mini-instruct,16,8.0,5,-32.760760397570486,-33.54188457270229,-47.35114928601293,-48.522835548710624,-22.2250388664238,-23.006163041555602,quadratic,quadratic,,,False,0.009381188118812271,-0.3749999999999994,0.1375
character_substitution,Phi-3.5-mini-instruct,32,2.0,6,-35.60455361354157,-36.02103467508546,-37.672036088193124,-38.29675768050896,-16.63590282585801,-17.052383887401902,quadratic,quadratic,0.03620138845518473,0.006790189635212771,False,0.7714438857585247,-0.14333345094879588,1.2414728795891055
character_substitution,Phi-3.5-mini-instruct,32,4.0,6,-49.08089748479006,-49.49737854633395,-48.06551973434438,-48.690241326660214,-26.741360165899245,-27.157841227443136,linear,linear,0.0104681170219736,0.000491145652045419,False,0.7173434360348284,0.13731060606060624,0.7617187500000001
character_substitution,Phi-3.5-mini-instruct,32,8.0,6,-52.15152728559241,-52.5680083471363,-50.83508524104607,-51.45980683336191,-32.569909532311655,-32.986390593855546,linear,linear,0.00036641799879640595,0.0001692229392835548,False,0.754134615384615,0.09722200332133232,1.032800427171829
character_substitution,Phi-3.5-mini-instruct,64,2.0,6,-31.20088621994264,-31.61736728148653,-36.76224349723808,-37.38696508955392,-16.51714821691195,-16.93362927845584,quadratic,quadratic,0.003171693063332667,0.015132224973907552,True,0.5602472594185804,-0.21522009966777633,0.7831122074445109
character_substitution,Phi-3.5-mini-instruct,64,4.0,6,-42.87896740328972,-43.29544846483361,-42.20826646323149,-42.832988055547325,-26.208882777727194,-26.625363839271085,linear,linear,0.009651107149845599,0.0008846080487136067,False,0.213241894718493,-0.32407576866764676,0.9397058823529411
character_substitution,Phi-3.5-mini-instruct,64,8.0,6,-47.253280224794565,-47.669761286338456,-51.39954733474513,-52.024268927060966,-32.56869529160852,-32.98517635315241,quadratic,quadratic,0.003911435517713898,0.0007772884765668526,False,0.36944871436556714,0.07536764705882354,0.5021067415730337
character_substitution,Qwen2.5-14B-Instruct,16,2.0,5,-21.30666274522673,-22.087786920358532,-27.539769661464547,-28.711455924162244,-8.467239760809614,-9.248363935941413,quadratic,quadratic,,,False,0.7747806262904335,-0.024038461538461387,0.8732529161116016
character_substitution,Qwen2.5-14B-Instruct,16,4.0,5,-32.92486181373087,-33.70598598886267,-38.12214213157699,-39.29382839427469,-12.859104255364514,-13.640228430496313,quadratic,quadratic,,,False,-0.8976102605866783,,
character_substitution,Qwen2.5-14B-Instruct,16,8.0,5,-21.219520872600558,-22.00064504773236,-28.47619233397976,-29.647878596677458,-14.276122477977129,-15.057246653108928,quadratic,quadratic,,,False,0.08742488733099643,-0.34221311475409905,0.5942877509279954
character_substitution,Qwen2.5-14B-Instruct,32,2.0,6,-25.798645916181563,-26.215126977725454,-37.71259058671649,-38.337312179032324,-12.809730931035517,-13.226211992579408,quadratic,quadratic,0.0023892015186948085,0.04134985875249598,True,0.7515734050516777,-0.06178032760298429,0.8807593856655287
character_substitution,Qwen2.5-14B-Instruct,32,4.0,6,-55.03135814992111,-55.447839211465,-53.41745051711347,-54.04217210942931,-18.815880368675607,-19.2323614302195,linear,linear,0.003342555130098072,0.00014191402286396773,False,-16.1113271208226,,
character_substitution,Qwen2.5-14B-Instruct,32,8.0,6,-31.65838304498444,-32.07486410652833,-37.456057781195504,-38.08077937351134,-19.52011704175057,-19.93659810329446,quadratic,quadratic,0.06480934415733708,0.01644533131820095,False,-0.08828530169176341,-0.3906249999999967,0.7039473684210535
character_substitution,Qwen2.5-14B-Instruct,64,2.0,6,-22.114316701693404,-22.530797763237295,-39.967820030903155,-40.59254162321899,-13.75843049224924,-14.17491155379313,quadratic,quadratic,0.013898821613314597,0.08444298295109513,True,0.6320271234769679,-0.17587528580760967,0.9333720847889722
character_substitution,Qwen2.5-14B-Instruct,64,4.0,6,-42.84800400138923,-43.26448506293312,-44.046501455172134,-44.67122304748797,-20.94974197206343,-21.36622303360732,quadratic,quadratic,0.0034465273912077103,0.0013501633281371506,False,1.9910191102001424,,
character_substitution,Qwen2.5-14B-Instruct,64,8.0,6,-36.94073404775697,-37.35721510930086,-35.47067732301314,-36.09539891532898,-24.38768780253127,-24.804168864075162,linear,linear,0.09193761449546077,0.0032233688741945474,False,-0.8495351323828937,,
character_substitution,Qwen2.5-32B-Instruct,16,2.0,5,-24.06173044012702,-24.84285461525882,-34.49130195172714,-35.662988214424836,-7.867161249403921,-8.64828542453572,quadratic,quadratic,,,False,0.7957837090599744,0.7499999999999999,1.0116525423728806
character_substitution,Qwen2.5-32B-Instruct,16,4.0,5,-26.697829265081218,-27.47895344021302,-39.60743105262866,-40.77911731532636,-13.696301801939331,-14.47742597707113,quadratic,quadratic,,,False,1.2076645810663766,-0.2574999999999999,1.4000000000000028
character_substitution,Qwen2.5-32B-Instruct,16,8.0,5,-25.704845892001742,-26.485970067133543,-28.553213023718484,-29.724899286416182,-13.068148352699193,-13.849272527830992,quadratic,quadratic,,,False,-0.27651920505773236,-0.39934842249657027,0.6029411764705882
character_substitution,Qwen2.5-32B-Instruct,32,2.0,6,-25.154225581598574,-25.570706643142465,-35.56551704273802,-36.19023863505386,-12.309533190253994,-12.726014251797885,quadratic,quadratic,0.004204592679411196,0.04462171964754139,True,0.7343003243057636,-0.024323878627968427,0.9089491362763911
character_substitution,Qwen2.5-32B-Instruct,32,4.0,6,-33.88732806591494,-34.303809127458834,-38.3557574486731,-38.98047904098894,-18.430495215125852,-18.846976276669743,quadratic,quadratic,0.001616532480138797,0.009065785468782908,True,1.413142692412047,-0.27250307532172685,1.4916109327826814
character_substitution,Qwen2.5-32B-Instruct,32,8.0,6,-34.87136328094701,-35.2878443424909,-34.381619457642365,-35.0063410499582,-18.92837248672631,-19.3448535482702,linear,linear,0.09796201847849745,0.0058584494027832,False,-1.2709051724137954,,
character_substitution,Qwen2.5-32B-Instruct,64,2.0,6,-22.773736660791876,-23.190217722335767,-40.5356313776407,-41.160352969956534,-13.901590636567882,-14.318071698111773,quadratic,quadratic,0.002427667211146839,0.07235026388290071,True,0.648437395755343,0.2629504504504505,0.739929687213669
character_substitution,Qwen2.5-32B-Instruct,64,4.0,6,-26.16345660689921,-26.5799376684431,-47.849567333023735,-48.47428892533957,-20.919992562866803,-21.336473624410694,quadratic,quadratic,0.0005969249254382318,0.04255092233618562,True,0.7489227416598715,0.01856884057971034,1.2501919274798676
character_substitution,Qwen2.5-32B-Instruct,64,8.0,6,-33.52994652784734,-33.946427589391234,-31.838828226249973,-32.46354981856581,-22.885968381438815,-23.302449442982706,linear,linear,0.1671133983338449,0.0034368161315321936,False,2.4601633375775886,,
character_substitution,Qwen2.5-7B-Instruct,16,2.0,5,-22.870756055888315,-23.651880231020115,-30.500461636416397,-31.672147899114094,-9.733383187650187,-10.514507362781986,quadratic,quadratic,,,False,0.8557769411304517,-0.2439516129032268,1.0965909090909078
character_substitution,Qwen2.5-7B-Instruct,16,4.0,5,-22.33115142774964,-23.11227560288144,-42.80600946609232,-43.97769572879002,-12.358519609866,-13.1396437849978,quadratic,quadratic,,,False,0.15425421977264897,0.07301051051051072,0.23161764705882368
character_substitution,Qwen2.5-7B-Instruct,16,8.0,5,-20.82964430797914,-21.61076848311094,-28.65253753208701,-29.824223794784707,-15.442420086033845,-16.223544261165646,quadratic,quadratic,,,False,0.16629907804800495,-0.15178571428571494,0.43229122834905753
character_substitution,Qwen2.5-7B-Instruct,32,2.0,6,-25.154428615667193,-25.570909677211084,-26.622447644027403,-27.24716923634324,-12.970299547490768,-13.386780609034659,quadratic,quadratic,0.40221103853631596,0.0394598779963913,False,0.8678302417157312,-0.04627214463668674,1.398203007375677
character_substitution,Qwen2.5-7B-Instruct,32,4.0,6,-41.676190305291996,-42.09267136683589,-39.68745036970063,-40.312171962016464,-20.383341574797583,-20.799822636341474,linear,linear,0.015048084595006742,0.0008602221444002517,False,13.256412337662107,,
character_substitution,Qwen2.5-7B-Instruct,32,8.0,6,-31.367231011021495,-31.783712072565386,-38.59437356724621,-39.219095159562045,-22.49702859090152,-22.913509652445413,quadratic,quadratic,0.055149267079277385,0.017826117480329137,False,0.06687848598884473,-0.2910995702005733,1.1041666666666652
character_substitution,Qwen2.5-7B-Instruct,64,2.0,6,-28.931895400976018,-29.34837646251991,-36.53267529361276,-37.1573968859286,-14.696867688449036,-15.113348749992927,quadratic,quadratic,0.042243787982642646,0.025626331290212453,False,0.6587189882905177,-0.335018136912184,0.8360242290748905
character_substitution,Qwen2.5-7B-Instruct,64,4.0,6,-34.23946611219087,-34.65594717373476,-37.62564201347038,-38.25036360578622,-21.370049037580706,-21.786530099124597,quadratic,quadratic,0.06165482358081101,0.009784779516381096,False,0.3522798887005233,-0.2653601694915284,0.5468749999999907
character_substitution,Qwen2.5-7B-Instruct,64,8.0,6,-36.19338351130512,-36.60986457284901,-38.52898952383522,-39.15371111615106,-26.24192627204522,-26.65840733358911,quadratic,quadratic,0.05157273190041082,0.006625516425875653,False,0.19771304622112312,-0.11510949840283624,0.49774618585298197
random_permutation,Mistral-7B-Instruct-v0.3,16,2.0,5,-44.40269666398945,-45.18382083912125,-50.92551862520817,-52.09720488790587,-9.585101448244313,-10.366225623376112,quadratic,quadratic,,,False,-1.580443280302614,,
random_permutation,Mistral-7B-Instruct-v0.3,16,4.0,5,-40.268266105403534,-41.049390280535334,-40.027938050658406,-41.199624313356104,-15.63631409975931,-16.41743827489111,linear,quadratic,,,False,-0.8056474480151199,,
random_permutation,Mistral-7B-Instruct-v0.3,16,8.0,5,-27.37909414733544,-28.16021832246724,-37.01351646019614,-38.18520272289384,-17.078412228801046,-17.859536403932847,quadratic,quadratic,,,False,0.09583333333333334,-0.1464311334289831,0.4375000000000001
random_permutation,Mistral-7B-Instruct-v0.3,32,2.0,6,-31.877468341377742,-32.29394940292163,-34.263770516876974,-34.88849210919281,-14.611735670700575,-15.028216732244466,quadratic,quadratic,0.10969706330279777,0.013704578894364378,False,0.3020423514947587,-0.09965610047847011,0.545018281535649
random_permutation,Mistral-7B-Instruct-v0.3,32,4.0,6,-44.45428240380062,-44.87076346534451,-46.960661772172145,-47.58538336448798,-21.92569371343272,-22.342174774976613,quadratic,quadratic,0.0003938509848629667,0.0013756019445613679,True,0.14325621799561095,-0.2814942993279536,0.9139150943396234
random_permutation,Mistral-7B-Instruct-v0.3,32,8.0,6,-48.16041096687371,-48.5768920284176,-52.69697707683119,-53.321698669147025,-25.987448745877902,-26.403929807421793,quadratic,quadratic,0.0012279473886478499,0.0009136399190937909,False,-0.5353308712920452,,
random_permutation,Mistral-7B-Instruct-v0.3,64,2.0,6,-35.97689162278044,-36.39337268432433,-47.2233086373211,-47.84803022963693,-15.18932174547913,-15.605802807023021,quadratic,quadratic,0.0023404187832464477,0.007219430811942294,True,0.41006160075067777,-0.0025501764446404984,0.5019968051118211
random_permutation,Mistral-7B-Instruct-v0.3,64,4.0,6,-35.48826604867603,-35.90474711021992,-40.61039065072157,-41.235112243037406,-21.8723990713874,-22.28888013293129,quadratic,quadratic,0.03992617828078705,0.005644540027866195,False,0.4658636174819366,0.25416421505237324,0.503225806451613
random_permutation,Mistral-7B-Instruct-v0.3,64,8.0,6,-50.74988480018857,-51.16636586173246,-54.96407323339096,-55.5887948257068,-28.88229873939369,-29.29877980093758,quadratic,quadratic,0.002257189391626391,0.0004261287509351675,False,-0.13494127516779186,-0.4876801390963392,0.3046874999999999
random_permutation,Phi-3.5-mini-instruct,16,2.0,5,-24.946765643111014,-25.727889818242815,-24.178006783719567,-25.349693046417265,-10.365542708707057,-11.146666883838856,linear,linear,,,False,0.6235714285714286,0.04222972972972943,0.7309633027522929
random_permutation,Phi-3.5-mini-instruct,16,4.0,5,-29.820473480956608,-30.60159765608841,-27.830569165389072,-29.00225542808677,-16.446136768159512,-17.227260943291313,linear,linear,,,False,4.049999999999997,,
random_permutation,Phi-3.5-mini-instruct,16,8.0,5,-30.52731493209201,-31.308439107223812,-32.23783062728185,-33.40951688997955,-21.11620962647119,-21.89733380160299,quadratic,quadratic,,,False,-0.2540681827822604,-0.41249999999999815,0.43750000000000017
random_permutation,Phi-3.5-mini-instruct,32,2.0,6,-35.691388318725195,-36.107869380269086,-38.26945377957351,-38.89417537188935,-16.515902592467324,-16.932383654011215,quadratic,quadratic,0.03300970879617582,0.006943106004202406,False,0.7664653207899965,-0.20350726115801973,1.1631885464693064
random_permutation,Phi-3.5-mini-instruct,32,4.0,6,-55.704223130416054,-56.120704191959945,-53.81658214417842,-54.441303736494255,-24.22565669571781,-24.6421377572617,linear,linear,0.0001129903092031879,9.209642209650425e-05,False,1.7722383720930037,,
random_permutation,Phi-3.5-mini-instruct,32,8.0,6,-46.88405313266992,-47.30053419421381,-54.92480178104611,-55.54952337336195,-29.14510288905177,-29.56158395059566,quadratic,quadratic,0.0028268260494389874,0.0013383382010465489,False,0.05967441020702925,-0.4641901549874812,0.5796124959913719
random_permutation,Phi-3.5-mini-instruct,64,2.0,6,-31.20088621994264,-31.61736728148653,-36.76224349723808,-37.38696508955392,-16.51714821691195,-16.93362927845584,quadratic,quadratic,0.003171693063332667,0.015132224973907552,True,0.5602472594185804,-0.21522009966777633,0.7831122074445109
random_permutation,Phi-3.5-mini-instruct,64,4.0,6,-45.65754331559121,-46.0740243771351,-43.661898182885665,-44.2866197752015,-25.98025926318681,-26.3967403247307,linear,linear,0.009170832122569908,0.0004199425982893128,False,-0.024159090909159284,-0.2604166666666649,1.0678501945524805
random_permutation,Phi-3.5-mini-instruct,64,8.0,6,-44.38234209680655,-44.798823158350444,-49.93052671818688,-50.55524831050272,-32.02143334272959,-32.43791440427348,quadratic,quadratic,0.004436049061049713,0.0018661402235862084,False,0.23024281698938148,-0.2880808362016612,0.4786184210526318
random_permutation,Qwen2.5-14B-Instruct,16,2.0,5,-21.30666274522673,-22.087786920358532,-27.539769661464547,-28.711455924162244,-8.467239760809614,-9.248363935941413,quadratic,quadratic,,,False,0.7747806262904335,-0.024038461538461387,0.8732529161116016
random_permutation,Qwen2.5-14B-Instruct,16,4.0,5,-25.91940534904498,-26.70052952417678,-28.786615955224065,-29.958302217921762,-12.37297751998739,-13.15410169511919,quadratic,quadratic,,,False,-0.22152238110040426,-0.313194444444444,0.57683030646992
random_permutation,Qwen2.5-14B-Instruct,16,8.0,5,-29.740389536512772,-30.521513711644573,-40.17900176505582,-41.35068802775352,-14.632817686495514,-15.413941861627313,quadratic,quadratic,,,False,-0.36664634413102276,-0.4122503823338756,1.0781250000000009
random_permutation,Qwen2.5-14B-Instruct,32,2.0,6,-25.798645916181563,-26.215126977725454,-37.71259058671649,-38.337312179032324,-12.809730931035517,-13.226211992579408,quadratic,quadratic,0.0023892015186948085,0.04134985875249598,True,0.7515734050516777,-0.06178032760298429,0.8807593856655287
random_permutation,Qwen2.5-14B-Instruct,32,4.0,6,-41.12110131581611,-41.53758237736,-40.647217738230594,-41.27193933054643,-18.099109777976537,-18.515590839520428,linear,linear,0.03920661960146756,0.002142436622869296,False,-1.650616945191904,,
random_permutation,Qwen2.5-14B-Instruct,32,8.0,6,-42.6824459330291,-43.09892699457299,-40.99973872829173,-41.62446032060757,-22.05798066412016,-22.47446172566405,linear,linear,0.026753969955619943,0.0007439882608910455,False,5.789245116200728,,
random_permutation,Qwen2.5-14B-Instruct,64,2.0,6,-22.114316701693404,-22.530797763237295,-39.967820030903155,-40.59254162321899,-13.75843049224924,-14.17491155379313,quadratic,quadratic,0.013898821613314597,0.08444298295109513,True,0.6320271234769679,-0.17587528580760967,0.9333720847889722
random_permutation,Qwen2.5-14B-Instruct,64,4.0,6,-44.99982298452722,-45.41630404607111,-43.4308806809419,-44.05560227325773,-20.987861574384084,-21.404342635927975,linear,linear,0.019741401689756393,0.0005096015612055432,False,3.5672972794506093,,
random_permutation,Qwen2.5-14B-Instruct,64,8.0,6,-36.90794021898784,-37.32442128053173,-41.73328531067309,-42.358006902988926,-27.483588347814468,-27.90006940935836,quadratic,quadratic,0.029016739521315806,0.004525115961923559,False,0.9464625007164554,0.2674780490738511,1.2508859930874203
random_permutation,Qwen2.5-32B-Instruct,16,2.0,5,-24.06173044012702,-24.84285461525882,-34.49130195172714,-35.662988214424836,-7.867161249403921,-8.64828542453572,quadratic,quadratic,,,False,0.7957837090599744,0.7499999999999999,1.0116525423728806
random_permutation,Qwen2.5-32B-Instruct,16,4.0,5,-26.485101971708893,-27.266226146840694,-26.93325130380446,-28.10493756650216,-12.364994046476522,-13.14611822160832,quadratic,quadratic,,,False,1.5439220183486229,,
random_permutation,Qwen2.5-32B-Instruct,16,8.0,5,-29.518544698827114,-30.299668873958915,-30.365866369893112,-31.53755263259081,-13.211020153796344,-13.992144328928143,quadratic,quadratic,,,False,-0.8659933333333315,,
random_permutation,Qwen2.5-32B-Instruct,32,2.0,6,-25.154225581598574,-25.570706643142465,-35.56551704273802,-36.19023863505386,-12.309533190253994,-12.726014251797885,quadratic,quadratic,0.004204592679411196,0.04462171964754139,True,0.7343003243057636,-0.024323878627968427,0.9089491362763911
random_permutation,Qwen2.5-32B-Instruct,32,4.0,6,-38.79580273495134,-39.21228379649523,-41.13557447923175,-41.760296071547586,-16.913030774009876,-17.329511835553767,quadratic,quadratic,0.001472343499233889,0.0032833170319077586,True,1.960940028511983,,
random_permutation,Qwen2.5-32B-Instruct,32,8.0,6,-40.15955489424214,-40.57603595578603,-38.2378465848022,-38.862568177118035,-18.70757924573891,-19.1240603072828,linear,linear,0.057084755950025136,0.0014292726784977993,False,-10.513785364526644,,
random_permutation,Qwen2.5-32B-Instruct,64,2.0,6,-22.773736660791876,-23.190217722335767,-40.5356313776407,-41.160352969956534,-13.901590636567882,-14.318071698111773,quadratic,quadratic,0.002427667211146839,0.07235026388290071,True,0.648437395755343,0.2629504504504505,0.739929687213669
random_permutation,Qwen2.5-32B-Instruct,64,4.0,6,-28.68751638928717,-29.10399745083106,-58.15678293989521,-58.781504532211045,-20.337659173901066,-20.754140235444957,quadratic,quadratic,0.0015284266522568634,0.027460355798362604,True,0.7930533835866191,0.4981162352646188,0.8049016631688771
random_permutation,Qwen2.5-32B-Instruct,64,8.0,6,-28.671602150656426,-29.088083212200317,-33.674767734859145,-34.29948932717498,-26.2051842802613,-26.62166534180519,quadratic,quadratic,0.11840524278360215,0.01791933503944395,False,0.8069205631445178,0.2683576814086017,1.124710720151297
random_permutation,Qwen2.5-7B-Instruct,16,2.0,5,-22.146738036536693,-22.927862211668494,-26.87593084099956,-28.047617103697256,-9.734887685509918,-10.516011860641717,quadratic,quadratic,,,False,0.8581060606060605,-0.3682692307692327,1.2483186709320921
random_permutation,Qwen2.5-7B-Instruct,16,4.0,5,-29.364998606159737,-30.146122781291538,-33.350072447307184,-34.52175871000488,-13.759160704173691,-14.54028487930549,quadratic,quadratic,,,False,-0.24918184694019588,-0.3437500000000008,0.20905172413793116
random_permutation,Qwen2.5-7B-Instruct,16,8.0,5,-26.4540779753898,-27.2352021505216,-36.10183388924827,-37.27352015194597,-17.346783097817607,-18.127907272949408,quadratic,quadratic,,,False,-0.13549210524394242,-0.4177631578947367,0.5551470588235298
random_permutation,Qwen2.5-7B-Instruct,32,2.0,6,-24.924906913544973,-25.341387975088864,-28.824297566381347,-29.449019158697183,-13.62960723729487,-14.046088298838761,quadratic,quadratic,0.2850420572411098,0.04766772389575472,False,0.7891829461457235,-0.07787431605304701,1.1699814724362014
random_permutation,Qwen2.5-7B-Instruct,32,4.0,6,-39.450574241602844,-39.867055303146735,-37.50908321676584,-38.133804809081674,-20.745997596678965,-21.162478658222856,linear,linear,0.015070287581965365,0.0012358761180382,False,5.7156083283935795,,
random_permutation,Qwen2.5-7B-Instruct,32,8.0,6,-32.388737338982324,-32.805218400526215,-39.97202374202229,-40.596745334338124,-22.555795377171698,-22.97227643871559,quadratic,quadratic,0.040312978028245856,0.015038185194817614,False,0.03358955278060049,-0.477308796093899,0.5294594645267752
random_permutation,Qwen2.5-7B-Instruct,64,2.0,6,-28.434825977147767,-28.851307038691658,-34.67472814396956,-35.299449736285396,-14.700548073075069,-15.11702913461896,quadratic,quadratic,0.08485833455437515,0.02787846275884241,False,0.6700735652704485,-0.3956139301101,0.8385083967026469
random_permutation,Qwen2.5-7B-Instruct,64,4.0,6,-46.97193971027294,-47.38842077181683,-45.802549317085465,-46.4272709094013,-23.387956107791535,-23.804437169335426,linear,linear,0.005957959571836279,0.0005786404793693771,False,-0.6090919452887575,,
random_permutation,Qwen2.5-7B-Instruct,64,8.0,6,-42.743760552030736,-43.16024161357463,-42.60173565770654,-43.22645725002238,-27.15859678483487,-27.57507784637876,linear,quadratic,0.019766698324634513,0.0016079825429275395,False,-0.4236977125542282,-0.488999484098413,1.2595893501805056
conditional_shift,Mistral-7B-Instruct-v0.3,16,2.0,5,-30.9106686020084,-31.6917927771402,-40.87647309610096,-42.04815935879866,-11.165710740683096,-11.946834915814895,quadratic,quadratic,,,False,0.04602754237288167,-0.21831510934393572,0.4657258064516131
conditional_shift,Mistral-7B-Instruct-v0.3,16,4.0,5,-28.930199138033842,-29.711323313165643,-31.820718045839655,-32.99240430853735,-14.839069914139941,-15.62019408927174,quadratic,quadratic,,,False,0.21979865771812082,0.1721761886765616,0.48338150289017345
conditional_shift,Mistral-7B-Instruct-v0.3,16,8.0,5,-34.50115968189275,-35.28228385702455,-39.673674628432934,-40.84536089113063,-17.28275360990343,-18.06387778503523,quadratic,quadratic,,,False,-0.044375414181576874,-0.4671816914498117,0.47426470588235276
conditional_shift,Mistral-7B-Instruct-v0.3,32,2.0,6,-34.91261402888579,-35.32909509042968,-40.508591513007204,-41.13331310532304,-14.967858758804802,-15.384339820348693,quadratic,quadratic,0.001345032490242425,0.007993846918859073,True,0.10281590812520652,-0.17927386109240467,0.5354166666666669
conditional_shift,Mistral-7B-Instruct-v0.3,32,4.0,6,-33.09391097116527,-33.51039203270916,-41.28598402653965,-41.91070561885549,-20.873273145287577,-21.289754206831468,quadratic,quadratic,0.03621855041323691,0.01357853749272508,False,0.30638121946598046,-0.17526854083786558,0.4118174687938055
conditional_shift,Mistral-7B-Instruct-v0.3,32,8.0,6,-41.26310334914494,-41.67958441068883,-39.26522712248942,-39.88994871480526,-27.25913415578698,-27.675615217330872,linear,linear,0.02702679737992968,0.0009439845455816424,False,-15.508325812275277,,
conditional_shift,Mistral-7B-Instruct-v0.3,64,2.0,6,-32.32183048836071,-32.7383115499046,-30.61112407287701,-31.235845665192848,-16.3261714752173,-16.74265253676119,linear,linear,0.210176096211762,0.003892275125309463,False,-0.1848161218424966,-0.22203788848910555,1.1588101295120061
conditional_shift,Mistral-7B-Instruct-v0.3,64,4.0,6,-34.95441472939516,-35.37089579093905,-59.60276803201873,-60.22748962433457,-22.552489832092856,-22.968970893636747,quadratic,quadratic,0.0005124907314471634,0.009970807584490315,True,0.4640129028251291,-0.1577824283816464,0.7870800266701371
conditional_shift,Mistral-7B-Instruct-v0.3,64,8.0,6,-40.42637725528217,-40.84285831682606,-44.13397416390134,-44.75869575621718,-29.618809344176995,-30.035290405720886,quadratic,quadratic,0.020931719758891326,0.0035428785071397947,False,0.4639373711444394,0.08387550121864439,0.6597222222222222
conditional_shift,Phi-3.5-mini-instruct,16,2.0,5,-23.69225061865068,-24.47337479378248,-28.010713894730046,-29.182400157427743,-11.109353587659529,-11.890477762791328,quadratic,quadratic,,,False,0.46310240963855415,-0.02319587628866003,0.7500000000000002
conditional_shift,Phi-3.5-mini-instruct,16,4.0,5,-34.75317913371443,-35.53430330884623,-35.838307287226364,-37.00999354992406,-16.908228017181102,-17.689352192312903,quadratic,quadratic,,,False,-0.37386363636363745,-0.4531249999999978,1.2291666666666536
conditional_shift,Phi-3.5-mini-instruct,16,8.0,5,-37.99705077879926,-38.778174953931064,-37.10777378805085,-38.27946005074855,-18.14556660589166,-18.92669078102346,linear,linear,,,False,2.1454044117647095,,
conditional_shift,Phi-3.5-mini-instruct,32,2.0,6,-28.108242804675704,-28.524723866219595,-37.278679462225256,-37.90340105454109,-16.157341832809056,-16.573822894352947,quadratic,quadratic,0.07052082234643803,0.03149699114679692,False,0.4591055555555556,-0.12843586355476555,0.5133916480438531
conditional_shift,Phi-3.5-mini-instruct,32,4.0,6,-39.2678274813034,-39.68430854284729,-37.370967693509826,-37.99568928582566,-24.184654643997405,-24.601135705541296,linear,linear,0.04785578736680746,0.0016030904383955585,False,-1.629708149779744,,
conditional_shift,Phi-3.5-mini-instruct,32,8.0,6,-41.07383442999467,-41.49031549153856,-43.01067987973306,-43.6354014720489,-27.1991069715621,-27.61558803310599,quadratic,quadratic,0.006869136120039415,0.0018312285799002032,False,0.9396443716268307,-0.20529390934843636,1.3177401742563986
conditional_shift,Phi-3.5-mini-instruct,64,2.0,6,-30.580204921343793,-30.996685982887684,-29.817349688665175,-30.44207128098101,-16.737417473389353,-17.153898534933244,linear,linear,0.17305858758587758,0.005987118655412296,False,0.4458412695491246,-0.171683482810165,1.113804220094727
conditional_shift,Phi-3.5-mini-instruct,64,4.0,6,-52.15211450061687,-52.56859556216076,-50.23214398614502,-50.856865578460855,-25.224804319516835,-25.641285381060726,linear,linear,0.0003312976138223649,0.00015780432517244444,False,-3.0970394736843567,,
conditional_shift,Phi-3.5-mini-instruct,64,8.0,6,-43.65498453177517,-44.07146559331906,-43.953964598965236,-44.57868619128107,-30.905493883485207,-31.321974945029098,quadratic,quadratic,0.01098007884060074,0.0008803696832416034,False,0.8219824419629739,0.0026254541326065704,1.4060543928216565
conditional_shift,Qwen2.5-14B-Instruct,16,2.0,5,-27.39065754978948,-28.17178172492128,-28.508895327365458,-29.680581590063156,-9.392694783017392,-10.173818958149191,quadratic,quadratic,,,False,-0.313104670558799,-0.3131046705587997,0.2608240223463687
conditional_shift,Qwen2.5-14B-Instruct,16,4.0,5,-27.114348352423235,-27.895472527555036,-33.87124485404723,-35.042931116744924,-13.950436915353244,-14.731561090485043,quadratic,quadratic,,,False,-0.2553115531522115,-0.4115853658536579,1.1079545454545439
conditional_shift,Qwen2.5-14B-Instruct,16,8.0,5,-35.77186917516545,-36.55299335029725,-33.771877801892316,-34.943564064590014,-14.163822005827292,-14.944946180959091,linear,linear,,,False,-607.7624999998077,,
conditional_shift,Qwen2.5-14B-Instruct,32,2.0,6,-29.684960177880832,-30.101441239424723,-32.36082425775171,-32.985545850067545,-11.98773861663512,-12.404219678179011,quadratic,quadratic,0.08121429943961965,0.013722029678405079,False,-0.03267455153007404,-0.368750000000002,0.4011286039453718
conditional_shift,Qwen2.5-14B-Instruct,32,4.0,6,-34.172190267819424,-34.588671329363315,-43.24228831709613,-43.86700990941197,-18.307017889202935,-18.723498950746826,quadratic,quadratic,0.00885882820191806,0.00907998892589581,True,-0.2888742006886378,-0.47007190725627235,0.32207749562171606
conditional_shift,Qwen2.5-14B-Instruct,32,8.0,6,-36.167613823439495,-36.58409488498339,-35.83995999646869,-36.46468158878453,-22.500269301894697,-22.916750363438588,linear,linear,0.08558007881137757,0.0025374735461224165,False,1.3256664265129672,0.3651574803149606,1.3769308656758474
conditional_shift,Qwen2.5-14B-Instruct,64,2.0,6,-29.97879054658371,-30.3952716081276,-28.131931006015414,-28.75665259833125,-13.50828272170547,-13.924763783249361,linear,linear,0.2846334958429348,0.005585899005057963,False,-1.4225463581690656,,
conditional_shift,Qwen2.5-14B-Instruct,64,4.0,6,-40.72838622881095,-41.14486729035484,-39.064038423679484,-39.68876001599532,-21.326818607103725,-21.743299668647616,linear,linear,0.019524930073420006,0.0014083981221531198,False,-2.913413747790457,,
conditional_shift,Qwen2.5-14B-Instruct,64,8.0,6,-34.752437405153316,-35.16891846669721,-37.69794735786303,-38.322668950178866,-26.69734079828152,-27.11382185982541,quadratic,quadratic,0.05116717879228233,0.005679816344995939,False,0.7879297060218113,0.21580813589980236,1.068213434022258
conditional_shift,Qwen2.5-32B-Instruct,16,2.0,5,-30.252455382335867,-31.033579557467668,-34.93012977733296,-36.101816040030656,-8.885447878396475,-9.666572053528274,quadratic,quadratic,,,False,1.2379625442399493,-0.1293604651162794,1.4449999999999992
conditional_shift,Qwen2.5-32B-Instruct,16,4.0,5,-28.35396382762027,-29.13508800275207,-29.098936743414527,-30.270623006112224,-11.680929542396962,-12.462053717528761,quadratic,quadratic,,,False,1.3240976900866213,-0.32499999999999996,1.4031385281385256
conditional_shift,Qwen2.5-32B-Instruct,16,8.0,5,-27.62172791782279,-28.40285209295459,-30.983924570478166,-32.155610833175864,-12.358910317168785,-13.140034492300584,quadratic,quadratic,,,False,0.9191666666666667,-0.32812499999999967,1.4044117647058851
conditional_shift,Qwen2.5-32B-Instruct,32,2.0,6,-37.95044789446591,-38.3669289560098,-46.14491777396099,-46.76963936627683,-12.134455645931908,-12.550936707475799,quadratic,quadratic,0.005523349654640976,0.005635386800251024,True,1.2082380857370891,-0.11684782608695671,1.4387612970632424
conditional_shift,Qwen2.5-32B-Instruct,32,4.0,6,-50.84182986409464,-51.25831092563853,-52.311247276764306,-52.93596886908014,-16.68746418630087,-17.10394524784476,quadratic,quadratic,0.0002352156936781771,0.0004291194187299291,True,3.8220286885245986,,
conditional_shift,Qwen2.5-32B-Instruct,32,8.0,6,-27.280343414356544,-27.696824475900435,-32.51975648006794,-33.14447807238378,-20.47304849562925,-20.88952955717314,quadratic,quadratic,0.09221830204904889,0.023390275875533176,False,0.72820706851864,0.02430555555555556,0.8680347700851995
conditional_shift,Qwen2.5-32B-Instruct,64,2.0,6,-23.742988891333397,-24.159469952877288,-39.742627825611564,-40.3673494179274,-14.508333100579165,-14.924814162123056,quadratic,quadratic,0.012339699692389428,0.06454312966350463,True,0.686602115524634,0.24666784445667408,1.2000682275208283
conditional_shift,Qwen2.5-32B-Instruct,64,4.0,6,-29.753175054022392,-30.169656115566283,-35.04322861859325,-35.66795021090908,-21.381850380684934,-21.798331442228825,quadratic,quadratic,0.05069790461999648,0.01633104796523099,False,0.8317655505885688,0.2040441176470589,1.1213650396698156
conditional_shift,Qwen2.5-32B-Instruct,64,8.0,6,-23.915107221255866,-24.331588282799757,-32.324262152508126,-32.94898374482396,-24.077992480894324,-24.494473542438215,quadratic,quadratic,0.04092001053349629,0.05079342934621304,True,0.6232069816643161,0.18772428229665072,0.6450240772362197
conditional_shift,Qwen2.5-7B-Instruct,16,2.0,5,-41.87952960975734,-42.66065378488914,-41.05930318207679,-42.23098944477449,-11.100750589536364,-11.881874764668163,linear,linear,,,False,2.399703305657268,,
conditional_shift,Qwen2.5-7B-Instruct,16,4.0,5,-32.00010704951032,-32.78123122464212,-30.42400529370594,-31.595691556403636,-15.992003422644242,-16.773127597776043,linear,linear,,,False,1.432800157977884,-0.26071428571428507,1.4888136836489718
conditional_shift,Qwen2.5-7B-Instruct,16,8.0,5,-31.08194960223159,-31.86307377736339,-36.51595447829148,-37.687640740989174,-18.644249677323614,-19.425373852455415,quadratic,quadratic,,,False,0.8362038766931338,-0.47010869565217234,1.0834103840682778
conditional_shift,Qwen2.5-7B-Instruct,32,2.0,6,-33.749758781719684,-34.166239843263575,-33.619183731969166,-34.243905324285,-17.21901690660621,-17.6354979681501,linear,quadratic,0.018461320972722194,0.004776204188533653,False,0.8738135132289987,-0.08302238805970093,1.315249182339106
conditional_shift,Qwen2.5-7B-Instruct,32,4.0,6,-40.36253781569514,-40.77901887723903,-39.179209562601336,-39.80393115491717,-23.554085294334214,-23.970566355878105,linear,linear,0.009484287810352227,0.00123879999571411,False,0.8733596785588129,0.013888888888888883,1.1470637962128034
conditional_shift,Qwen2.5-7B-Instruct,32,8.0,6,-35.45284141442099,-35.86932247596488,-35.5061690014721,-36.13089059378794,-26.99425687011593,-27.41073793165982,quadratic,quadratic,0.07798965075036206,0.0031502824617456113,False,0.7387126865671643,0.22176724137931028,1.4432845835397548
conditional_shift,Qwen2.5-7B-Instruct,64,2.0,6,-31.676386571219723,-32.092867632763614,-30.006891340957573,-30.63161293327341,-16.32337568778896,-16.73985674933285,linear,linear,0.058669242175618495,0.00624283944307989,False,0.6204918844022169,-0.198447677711694,1.1132352941176449
conditional_shift,Qwen2.5-7B-Instruct,64,4.0,6,-45.223557247858665,-45.640038309402556,-45.44611288344075,-46.070834475756584,-25.19806112068816,-25.61454218223205,quadratic,quadratic,0.015192796689991255,0.0011704838465393178,False,0.0795531242040907,-0.09843630667057901,1.362098016974626
conditional_shift,Qwen2.5-7B-Instruct,64,8.0,6,-36.691909036973804,-37.108390098517695,-38.500502119441556,-39.12522371175739,-29.3245019847512,-29.740983046295092,quadratic,quadratic,0.05515218183872064,0.003475133816423429,False,0.5444843981427732,0.21054687500000008,0.6808755760368663
composed_permutation,Mistral-7B-Instruct-v0.3,16,4.0,5,-33.690830492146,-34.4719546672778,-36.503723070783444,-37.67540933348114,-18.12814795992393,-18.90927213505573,quadratic,quadratic,,,False,0.6266960651289009,0.0001685825245842684,1.324999999999999
composed_permutation,Mistral-7B-Instruct-v0.3,16,8.0,5,-38.57651444659889,-39.35763862173069,-42.239972609369744,-43.41165887206744,-26.093937845220612,-26.875062020352413,quadratic,quadratic,,,False,0.269601372756072,0.062499999999999244,0.9080882352941178
composed_permutation,Mistral-7B-Instruct-v0.3,32,4.0,6,-38.68433226970036,-39.100813331244254,-53.2110248520264,-53.835746444342234,-23.279682804962142,-23.696163866506033,quadratic,quadratic,0.0033928634777626427,0.005421744777657957,True,0.5983695652173915,-0.40503233120591725,1.1388888888888913
composed_permutation,Mistral-7B-Instruct-v0.3,32,8.0,6,-48.35546677828551,-48.7719478398294,-63.56932355686776,-64.1940451491836,-32.024507203785674,-32.440988265329565,quadratic,quadratic,9.322637872416548e-05,0.001033146958754606,True,0.627033874403886,-0.0681442753568092,0.7951210968775058
composed_permutation,Mistral-7B-Instruct-v0.3,64,4.0,6,-36.37787458075296,-36.79435564229685,-43.91024837575287,-44.53496996806871,-23.72656851202679,-24.143049573570682,quadratic,quadratic,0.007151658343707156,0.0059283625327030106,False,0.5738157444706241,-0.16569767441860142,0.8817307692307687
composed_permutation,Mistral-7B-Instruct-v0.3,64,8.0,6,-44.627620648081574,-45.044101709625465,-52.72707704807155,-53.351798640387386,-32.42675311020991,-32.8432341717538,quadratic,quadratic,0.0010627713059905757,0.001568222332866574,True,0.5712903466826063,-0.007327586206896369,0.633470322308176
composed_permutation,Phi-3.5-mini-instruct,16,4.0,5,-33.03183690462808,-33.81296107975988,-32.014758521061964,-33.18644478375966,-21.03027974857296,-21.81140392370476,linear,linear,,,False,0.3578309692671395,0.14962121212121213,1.0312500000000002
composed_permutation,Phi-3.5-mini-instruct,16,8.0,5,-32.99778637844642,-33.77891055357822,-31.48453237775456,-32.65621864045226,-25.29259631713301,-26.073720492264812,linear,linear,,,False,0.5238562091503272,-0.03749999999999984,0.6519230769230769
composed_permutation,Phi-3.5-mini-instruct,32,4.0,6,-46.58495900563368,-47.00144006717757,-45.36018179346253,-45.98490338577837,-24.9361031766949,-25.35258423823879,linear,linear,0.003935018418380284,0.00039548827297311985,False,0.9363183890577521,0.06685225846925975,1.1708357395203493
composed_permutation,Phi-3.5-mini-instruct,32,8.0,6,-56.81141490065608,-57.22789596219997,-56.11339577673089,-56.738117369046726,-33.858932861120465,-34.275413922664356,linear,linear,0.0005646338688250564,9.658372965607789e-05,False,-0.05720505617977988,-0.44268747158443045,0.6272865853658536
composed_permutation,Phi-3.5-mini-instruct,64,4.0,6,-46.54739405556782,-46.96387511711171,-47.31437966646204,-47.939101258777875,-24.152104008656398,-24.56858507020029,quadratic,quadratic,0.00475621606725301,0.0006021176401980418,False,0.7359315361021652,0.03752022711632258,1.1008190108885674
composed_permutation,Phi-3.5-mini-instruct,64,8.0,6,-52.38823069738184,-52.804711758925734,-50.50474196274119,-51.12946355505703,-33.37866751440344,-33.79514857594733,linear,linear,0.005176268940458857,0.00017410903509875364,False,0.7751856435643575,-0.04687500000000081,0.8244960449780837
composed_permutation,Qwen2.5-14B-Instruct,16,4.0,5,-37.27888884692638,-38.06001302205818,-39.19381797521655,-40.365504237914244,-15.934243152526449,-16.71536732765825,quadratic,quadratic,,,False,0.8823445595854913,-0.23915289256198358,0.9795241735441804
composed_permutation,Qwen2.5-14B-Instruct,16,8.0,5,-40.13005894110974,-40.91118311624154,-39.45428567557185,-40.62597193826955,-26.885265767996625,-27.666389943128426,linear,linear,,,False,0.4705526315789478,0.18749999999999983,0.8615506329113923
composed_permutation,Qwen2.5-14B-Instruct,32,4.0,6,-56.621233900821636,-57.03771496236553,-63.36172961865773,-63.98645121097357,-22.566307074774826,-22.982788136318717,quadratic,quadratic,0.0008761660895037175,0.0002622209204717833,False,0.2919234523145578,-0.20940271145977984,0.4092594905094906
composed_permutation,Qwen2.5-14B-Instruct,32,8.0,6,-69.56286817365725,-69.97934923520114,-67.56317755081714,-68.18789914313298,-31.612399631013716,-32.02888069255761,linear,linear,5.9636305954163686e-05,8.372245073575258e-06,False,-94.74124999993752,,
composed_permutation,Qwen2.5-14B-Instruct,64,4.0,6,-48.03718244026625,-48.45366350181014,-60.93883029538914,-61.563551887704975,-24.053331804706012,-24.469812866249903,quadratic,quadratic,0.00044523875666701955,0.001117594255514009,True,0.37219311377245484,0.11762804661183975,0.7616150442477858
composed_permutation,Qwen2.5-14B-Instruct,64,8.0,6,-58.86994681618409,-59.28642787772798,-57.38037577865787,-58.00509737097371,-31.19431966024176,-31.61080072178565,linear,linear,0.002228253350662603,4.6000415923998435e-05,False,1.8115297833935284,,
composed_permutation,Qwen2.5-32B-Instruct,16,4.0,5,-28.139600362750215,-28.920724537882016,-26.932760933947435,-28.104447196645133,-17.57823273645613,-18.35935691158793,linear,linear,,,False,0.15220823341326967,-0.256944444444445,0.9471153846153842
composed_permutation,Qwen2.5-32B-Instruct,16,8.0,5,-34.852271671914714,-35.633395847046515,-35.42066353610994,-36.59234979880764,-23.68620138234896,-24.46732555748076,quadratic,quadratic,,,False,0.17518903591682253,-0.21875000000000036,1.1625
composed_permutation,Qwen2.5-32B-Instruct,32,4.0,6,-44.03991688449387,-44.45639794603776,-50.75117169886085,-51.37589329117669,-22.732395177437336,-23.148876238981227,quadratic,quadratic,0.002092848889703508,0.0019512797075895809,False,0.052787558685446684,-0.1636813883568948,0.9623161764705882
composed_permutation,Qwen2.5-32B-Instruct,32,8.0,6,-45.72749060662802,-46.14397166817191,-48.568555527395404,-49.19327711971124,-31.81054831133269,-32.22702937287658,quadratic,quadratic,0.005844104324650594,0.001322850200486114,False,0.2321486495432284,0.09106975179737978,0.858552631578947
composed_permutation,Qwen2.5-32B-Instruct,64,4.0,6,-44.63082008698302,-45.04730114852691,-45.68808539286671,-46.31280698518255,-23.986212290255203,-24.402693351799094,quadratic,quadratic,0.010065225691378377,0.0013757479484123568,False,0.1888083463782564,-0.05442415730336954,1.3921557890209295
composed_permutation,Qwen2.5-32B-Instruct,64,8.0,6,-53.47160360312769,-53.88808466467158,-54.393725267208026,-55.01844685952386,-32.78213476816141,-33.198615829705304,quadratic,quadratic,0.0013954026887817107,0.0002144185327262708,False,0.053631258508548746,-0.32500000000000034,0.4625
composed_permutation,Qwen2.5-7B-Instruct,16,4.0,5,-34.185609109415886,-34.966733284547686,-32.279209161019054,-33.45089542371675,-17.829920642951368,-18.61104481808317,linear,linear,,,False,3.246862139917713,,
composed_permutation,Qwen2.5-7B-Instruct,16,8.0,5,-43.37745859939981,-44.15858277453161,-41.99184078290797,-43.16352704560567,-25.279148271420404,-26.060272446552204,linear,linear,,,False,-0.7229338842975294,,
composed_permutation,Qwen2.5-7B-Instruct,32,4.0,6,-44.82345692431898,-45.23993798586287,-42.88984083420144,-43.51456242651728,-25.378064791392475,-25.794545852936366,linear,linear,0.016437023663564778,0.0004601946910565569,False,-2.0899608013937065,,
composed_permutation,Qwen2.5-7B-Instruct,32,8.0,6,-55.931511653016116,-56.34799271456001,-83.09489579120992,-83.71961738352576,-34.093372697553676,-34.50985375909757,quadratic,quadratic,1.1857442625058415e-06,0.0002963842278317084,True,0.7254166666666674,0.603924418604651,0.8711331265947774
composed_permutation,Qwen2.5-7B-Instruct,64,4.0,6,-38.962068947897464,-39.378550009441355,-41.64731166984306,-42.2720332621589,-24.477444425907695,-24.893925487451586,quadratic,quadratic,0.010924633056540517,0.003895810672319602,False,0.6030016087271066,-0.16799962204537178,1.163961780786443
composed_permutation,Qwen2.5-7B-Instruct,64,8.0,6,-49.94448932787861,-50.3609703894225,-60.101171479347656,-60.72589307166349,-33.161511874158876,-33.57799293570277,quadratic,quadratic,0.0006818852349289338,0.0006761172430505136,False,0.6263833583771602,0.2170454545454546,0.670263452914798
copy_previous,Mistral-7B-Instruct-v0.3,16,,5,-22.420223536306587,-23.201347711438387,-25.431662387143046,-26.603348649840747,-9.37441311146181,-10.155537286593608,quadratic,quadratic,,,False,0.4795178105994786,0.05288461538461544,0.8081896551724155
copy_previous,Mistral-7B-Instruct-v0.3,32,,6,-31.747208714016153,-32.163689775560044,-30.81784416214868,-31.442565754464518,-14.968842681217698,-15.385323742761589,linear,linear,0.19561066604149713,0.009131131613610887,False,0.5081912770809579,0.1381578947368421,0.6003870816597208
copy_previous,Mistral-7B-Instruct-v0.3,64,,6,-50.679404158479024,-51.095885220022915,-54.65297117611379,-55.277692768429624,-14.23967333491284,-14.656154396456731,quadratic,quadratic,0.0033109523474633418,0.0006436368497283013,False,0.7408232964663978,-0.056783707865169945,1.3964857866122624
copy_previous,Qwen2.5-14B-Instruct,16,,5,-20.927388657472584,-21.708512832604384,-33.21741931439825,-34.38910557709595,-4.739688406233784,-5.520812581365583,quadratic,quadratic,,,False,0.5911141346006439,0.5704225352112676,0.7569444444444446
copy_previous,Qwen2.5-14B-Instruct,32,,6,-27.113738086940877,-27.53021914848477,-47.69676224233251,-48.32148383464835,-9.603199526875601,-10.01968058841949,quadratic,quadratic,0.007143522479204359,0.03727166909091304,True,0.5212818060780254,-0.20690745617109102,0.8456938630221261
copy_previous,Qwen2.5-14B-Instruct,64,,6,-28.939197362610734,-29.355678424154625,-40.55829422020719,-41.183015812523024,-10.048364323984979,-10.464845385528868,quadratic,quadratic,0.024485699137493377,0.022934293128978416,False,0.5839973162463404,0.05719339622641481,0.6071843038925729
copy_previous,Qwen2.5-32B-Instruct,16,,5,-22.025355969464353,-22.806480144596154,-26.34447752345438,-27.516163786152077,-5.193948298090053,-5.975072473221852,quadratic,quadratic,,,False,0.5035227495587369,0.011363636363636331,0.6857876712328772
copy_previous,Qwen2.5-32B-Instruct,32,,6,-27.27163739081616,-27.68811845236005,-34.31917701422797,-34.943898606543804,-8.710522688797496,-9.127003750341387,quadratic,quadratic,0.10682867992549734,0.03494482233000467,False,0.5036255216577926,-0.1890202702702708,0.9231557377049187
copy_previous,Qwen2.5-32B-Instruct,64,,6,-23.77158823878611,-24.18806930033,-23.7493611328889,-24.374082725204737,-10.184689108391234,-10.601170169935124,linear,quadratic,0.351251598774827,0.038359895658376116,False,0.641173438102473,0.06442299358640999,0.880015432098766
copy_previous,Qwen2.5-7B-Instruct,16,,5,-31.494203562706133,-32.275327737837934,-35.50250345962942,-36.67418972232712,-11.025166026961102,-11.806290202092901,quadratic,quadratic,,,False,0.7097222203111379,-0.2864247311827966,1.170454545454545
copy_previous,Qwen2.5-7B-Instruct,32,,6,-34.91850817093134,-35.33498923247523,-35.26706380901421,-35.89178540133005,-15.870043723244656,-16.286524784788547,quadratic,quadratic,0.09400554673797774,0.006845846188803895,False,0.7300377136226176,0.07352941176470586,0.7945995145631068
copy_previous,Qwen2.5-7B-Instruct,64,,6,-38.64476311671214,-39.06124417825603,-37.827254933164205,-38.45197652548004,-15.283119116088205,-15.699600177632096,linear,linear,0.04978992014160958,0.0028628492015613364,False,1.1332348954298987,-0.4158163265306114,1.2465277777777743
running_max,Mistral-7B-Instruct-v0.3,16,2.0,5,-15.25210287987931,-16.03322705501111,-20.13633574982525,-21.308022012522947,-7.593186576929519,-8.374310752061318,quadratic,quadratic,,,False,0.48158119449595,0.01325757575757576,0.7124999999999991
running_max,Mistral-7B-Instruct-v0.3,16,4.0,5,-14.60122832551125,-15.38235250064305,-25.41815197651093,-26.589838239208632,-8.139658011703148,-8.920782186834947,quadratic,quadratic,,,False,0.4661869393465558,0.017454954954954974,0.5248404255319149
running_max,Mistral-7B-Instruct-v0.3,16,8.0,5,-17.829997539859427,-18.611121714991228,-21.234640481368437,-22.406326744066135,-12.408696585188288,-13.189820760320087,quadratic,quadratic,,,False,0.49735357334766245,0.06738826815642458,0.6322674418604654
running_max,Mistral-7B-Instruct-v0.3,32,2.0,6,-20.537222559337195,-20.953703620881086,-28.972633551262618,-29.597355143578454,-9.791626265384902,-10.208107326928793,quadratic,quadratic,0.2539353178254942,0.08186049105433386,False,0.5162325453820068,0.2977335164835163,0.5523952095808383
running_max,Mistral-7B-Instruct-v0.3,32,4.0,6,-16.30764948775063,-16.72413054929452,-23.287527486866345,-23.912249079182182,-7.407881635071856,-7.824362696615746,quadratic,quadratic,0.5155890716753303,0.15628455681525655,False,0.5440173564852703,-0.21818124301327377,0.5747750102299699
running_max,Mistral-7B-Instruct-v0.3,32,8.0,6,-19.635482410125462,-20.051963471669353,-21.46629458321354,-22.091016175529376,-10.16279163854283,-10.579272700086719,quadratic,quadratic,0.28920903202059384,0.06350109081868893,False,0.5457191684758561,0.023198689956332296,0.8555596013945665
running_max,Mistral-7B-Instruct-v0.3,64,2.0,6,-23.253612229676207,-23.6700932912201,-21.25797480727943,-21.882696399595268,-10.355014462650999,-10.77149552419489,linear,linear,0.9093818640069182,0.02126136971225404,False,1.0601990311710197,0.10272550942109317,1.0272230101280888
running_max,Mistral-7B-Instruct-v0.3,64,4.0,6,-20.368927561226606,-20.785408622770497,-20.794503242027517,-21.419224834343353,-8.178784662205146,-8.595265723749037,quadratic,quadratic,0.37908150686153336,0.04790502875341301,False,0.6045613269877977,-0.011582167832167652,0.7355818676411424
running_max,Mistral-7B-Instruct-v0.3,64,8.0,6,-17.36758495931959,-17.78406602086348,-15.388172483769555,-16.012894076085388,-8.487496206445037,-8.903977267988928,linear,linear,0.1823282682466747,0.04763984165923415,False,1.7632727722772288,,
running_max,Qwen2.5-14B-Instruct,16,2.0,5,-25.26040281689228,-26.04152699202408,-41.327958185279364,-42.49964444797706,-4.295054267983005,-5.076178443114804,quadratic,quadratic,,,False,0.47723120781940875,-0.06874999999999994,1.4943181818181677
running_max,Qwen2.5-14B-Instruct,16,4.0,5,-20.338289966214646,-21.119414141346446,-31.905905900635595,-33.07759216333329,-4.733258638242155,-5.514382813373954,quadratic,quadratic,,,False,0.43771420273017697,-0.003906250000001259,0.5350000000000004
running_max,Qwen2.5-14B-Instruct,16,8.0,5,-16.48500582018906,-17.26612999532086,-24.848563017843933,-26.020249280541634,-5.138990193704366,-5.920114368836165,quadratic,quadratic,,,False,0.43938106180716396,-0.3344594594594593,0.7746833084947857
running_max,Qwen2.5-14B-Instruct,32,2.0,6,-26.93098996062234,-27.347471022166232,-36.73395009945979,-37.358671691775626,-6.625532322204229,-7.042013383748119,quadratic,quadratic,0.06081674160448667,0.03807304498512593,False,0.5132402845457732,-0.27202041277700384,0.9680507184774954
running_max,Qwen2.5-14B-Instruct,32,4.0,6,-27.031863583443457,-27.448344644987348,-38.015132143761704,-38.63985373607754,-7.466752073196304,-7.883233134740194,quadratic,quadratic,0.02796541003116823,0.03688620039377741,True,0.4593003737514058,0.07117391151822063,0.5508957654723128
running_max,Qwen2.5-14B-Instruct,32,8.0,6,-25.124626797211985,-25.541107858755876,-27.194340594997527,-27.819062187313364,-8.994342697321464,-9.410823758865355,quadratic,quadratic,0.2989363598574431,0.040366170358751334,False,0.39531403661396486,-0.2335526315789472,1.4060576023247835
running_max,Qwen2.5-14B-Instruct,64,2.0,6,-40.3411744523447,-40.757655513888594,-42.37015945374671,-42.994881046062545,-6.997646572485706,-7.4141276340295965,quadratic,quadratic,0.0007493228753269753,0.0024826666418403926,True,0.6689321155078813,0.01406249999999993,0.8437500000000004
running_max,Qwen2.5-14B-Instruct,64,4.0,6,-32.77844477610434,-33.19492583764823,-45.57320692143293,-46.197928513748764,-6.975288906324316,-7.391769967868206,quadratic,quadratic,0.0006019080913059132,0.01335779162976522,True,0.5565620071225494,0.13448275862068973,0.5707831325301205
running_max,Qwen2.5-14B-Instruct,64,8.0,6,-22.380808648721107,-22.797289710264998,-23.007755598922508,-23.632477191238344,-8.40494417160428,-8.821425233148169,quadratic,quadratic,0.7427333137967107,0.05716644393390613,False,0.5527611605031685,0.09873417721518998,0.6202260746260756
running_max,Qwen2.5-32B-Instruct,16,2.0,5,-29.60003088282749,-30.38115505795929,-48.100127423133806,-49.2718136858315,-4.090507156016807,-4.871631331148606,quadratic,quadratic,,,False,0.5061763130412373,0.40312500000000007,1.1458333333333164
running_max,Qwen2.5-32B-Instruct,16,4.0,5,-20.662034407507644,-21.443158582639445,-22.00596230142518,-23.177648564122876,-4.0378142017731165,-4.818938376904915,quadratic,quadratic,,,False,0.45132316313823156,-0.040570175438596597,0.9025862068965516
running_max,Qwen2.5-32B-Instruct,16,8.0,5,-23.826151452889277,-24.60727562802108,-23.677580180692622,-24.84926644339032,-4.7051723369536695,-5.4862965120854685,linear,quadratic,,,False,0.3215632779874896,0.16525423728813474,1.4058577405857748
running_max,Qwen2.5-32B-Instruct,32,2.0,6,-36.75487425775028,-37.17135531929417,-42.58575971325993,-43.21048130557577,-6.756993860622172,-7.173474922166062,quadratic,quadratic,0.027881648233542552,0.004791019882835487,False,0.47149545132470927,0.27312646370023475,0.5078125
running_max,Qwen2.5-32B-Instruct,32,4.0,6,-25.345947931076857,-25.762428992620748,-25.168572690164375,-25.79329428248021,-6.5705906068569995,-6.98707166840089,linear,quadratic,0.5285263260485206,0.016109349399538998,False,0.39698483834612974,-0.01450742506461301,1.2092820044734645
running_max,Qwen2.5-32B-Instruct,32,8.0,6,-29.243099122915446,-29.659580184459337,-29.252073330522094,-29.87679492283793,-7.291790994374633,-7.708272055918523,quadratic,quadratic,0.03821138900233489,0.011560553147559447,False,0.3149998831010945,-0.3347042172628801,0.9361441798941791
running_max,Qwen2.5-32B-Instruct,64,2.0,6,-41.09098665875143,-41.507467720295324,-40.260487878348655,-40.88520947066449,-6.778682447870189,-7.195163509414079,linear,linear,0.03468143651138448,0.0011293803286222188,False,0.7722975660197258,0.16537001861916686,1.3072562441369797
running_max,Qwen2.5-32B-Instruct,64,4.0,6,-24.85839809573084,-25.274879157274732,-23.336434422196763,-23.961156014512596,-6.59005307446896,-7.00653413601285,linear,linear,0.7028503435038732,0.014508244658567903,False,0.4254110376207903,0.1462130594680273,0.8811559804240982
running_max,Qwen2.5-32B-Instruct,64,8.0,6,-24.145828712325844,-24.562309773869735,-22.364299808926567,-22.989021401242404,-6.988254247462782,-7.404735309006672,linear,linear,0.1945501414454445,0.01677643979642909,False,0.3140275262000259,0.13663630612332192,1.1817858290620813
running_max,Qwen2.5-7B-Instruct,16,2.0,5,-25.581930319548178,-26.36305449467998,-23.58682877791979,-24.75851504061749,-7.211962737438535,-7.993086912570334,linear,linear,,,False,4.828030303030271,,
running_max,Qwen2.5-7B-Instruct,16,4.0,5,-29.318113054506966,-30.099237229638767,-29.200189473792086,-30.371875736489784,-7.621842438745739,-8.402966613877538,linear,quadratic,,,False,0.6658825338253382,0.0624999999999998,0.7428441485834383
running_max,Qwen2.5-7B-Instruct,16,8.0,5,-19.248260390927015,-20.029384566058816,-20.89988349317656,-22.071569755874258,-9.331343529773218,-10.112467704905017,quadratic,quadratic,,,False,0.4586108197867505,-0.11742676431424787,0.5673732943469786
running_max,Qwen2.5-7B-Instruct,32,2.0,6,-31.761642932979072,-32.17812399452296,-31.511242436587338,-32.135964028903174,-9.17682768767324,-9.59330874921713,linear,linear,0.13954661800411447,0.010092449969011985,False,0.8222982686895879,-0.10785269201896593,0.8841193741149711
running_max,Qwen2.5-7B-Instruct,32,4.0,6,-31.581894621839204,-31.998375683383095,-37.59362915257089,-38.21835074488673,-8.294912040423828,-8.711393101967719,quadratic,quadratic,0.0038552871049022234,0.013511549982619274,True,0.6996199960567604,0.13427197802197796,0.8402364234570403
running_max,Qwen2.5-7B-Instruct,32,8.0,6,-27.149601220509936,-27.566082282053827,-33.580478357918764,-34.2051999502346,-10.19494206244066,-10.611423123984551,quadratic,quadratic,0.03499470594274833,0.032960837448459096,False,0.6181519332676871,0.053010182728098834,0.7509816753926705
running_max,Qwen2.5-7B-Instruct,64,2.0,6,-33.87984679481652,-34.29632785636041,-32.30583982713469,-32.93056141945053,-8.424412937324771,-8.840893998868662,linear,linear,0.15317850143905817,0.005026174714691425,False,1.6088229108244487,,
running_max,Qwen2.5-7B-Instruct,64,4.0,6,-26.565378422161793,-26.981859483705684,-36.474560229447604,-37.09928182176344,-7.504724456746697,-7.921205518290587,quadratic,quadratic,0.0052080257325833375,0.03463062521586615,True,0.6230812922977764,-0.016108195000705476,0.722997898894579
running_max,Qwen2.5-7B-Instruct,64,8.0,6,-29.696998151839722,-30.113479213383613,-37.40198951876822,-38.026711111084055,-8.259348919456977,-8.675829981000867,quadratic,quadratic,0.029064067707502597,0.022449404881797683,False,0.7240186387845676,-0.2689746884511574,0.8545363601310915
running_xor,Mistral-7B-Instruct-v0.3,16,,5,-30.199992613667156,-30.981116788798957,-37.21004290137942,-38.381729164077115,-11.60487504751505,-12.385999222646848,quadratic,quadratic,,,False,0.4041529435981886,-0.04464285714285658,0.5100255972696246
running_xor,Mistral-7B-Instruct-v0.3,32,,6,-33.56687095961689,-33.98335202116078,-39.87391851018725,-40.49864010250309,-15.801030858319546,-16.217511919863437,quadratic,quadratic,0.01100057555210457,0.009133556790714528,False,0.46740132801850554,0.025893739194862878,0.546875
running_xor,Mistral-7B-Instruct-v0.3,64,,6,-36.23086104578044,-36.647342107324334,-37.031282322657454,-37.65600391497329,-15.727654426807966,-16.144135488351857,quadratic,quadratic,0.03355136959860713,0.00326587444022022,False,0.5123231520070192,0.04314438695867217,0.6443345905172425
running_xor,Phi-3.5-mini-instruct,16,,5,-33.829863953633314,-34.610988128765115,-33.2665071100664,-34.4381933727641,-9.191470597176414,-9.972594772308213,linear,linear,,,False,0.24848484848484817,-0.27608267716535817,0.8671874999999999
running_xor,Phi-3.5-mini-instruct,32,,6,-41.46988721767476,-41.88636827921865,-40.958463482414274,-41.58318507473011,-13.63361524753902,-14.05009630908291,linear,linear,0.03113832180938079,0.0019276618021200342,False,0.48455188679245287,0.1577380952380952,0.6884982890850817
running_xor,Phi-3.5-mini-instruct,64,,6,-38.7071178437933,-39.12359890533719,-39.580937807246045,-40.20565939956188,-14.754112973214127,-15.170594034758018,quadratic,quadratic,0.025115323717612718,0.0023161360555172643,False,0.3583229537366545,0.018520562197336547,1.2375000000000025
running_xor,Qwen2.5-14B-Instruct,16,,5,-28.61738283818832,-29.39850701332012,-27.913468871972455,-29.085155134670153,-6.855364737220183,-7.636488912351982,linear,linear,,,False,0.5901041666666669,-0.3750000000000026,0.6507352941176471
running_xor,Qwen2.5-14B-Instruct,32,,6,-38.04024675032076,-38.45672781186465,-36.156489385879546,-36.78121097819538,-12.299885230174038,-12.716366291717929,linear,linear,0.07607812543475197,0.0016882308832137606,False,0.402558095952024,0.05835642545436348,0.7563933466784281
running_xor,Qwen2.5-14B-Instruct,64,,6,-35.177826618978976,-35.59430768052287,-33.72710027032495,-34.35182186264078,-13.095810483556697,-13.512291545100588,linear,linear,0.1213817345134663,0.002594849733081487,False,0.20622147894627618,-0.22087452978663288,1.354720891551207
running_xor,Qwen2.5-32B-Instruct,16,,5,-28.75235084834832,-29.533475023480122,-32.76889262127601,-33.94057888397371,-6.214011039865003,-6.995135214996802,quadratic,quadratic,,,False,0.3049124513618676,-0.36250000000000476,0.5121428571428572
running_xor,Qwen2.5-32B-Instruct,32,,6,-33.57710461778964,-33.99358567933353,-32.02896512735507,-32.65368671967091,-10.85928760292126,-11.27576866446515,linear,linear,0.16922931491972093,0.005529543833732995,False,0.10866362126245827,-0.2616478539413946,0.7665094339622641
running_xor,Qwen2.5-32B-Instruct,64,,6,-30.616379707223892,-31.032860768767783,-29.070316938180255,-29.69503853049609,-12.594496711070505,-13.010977772614396,linear,linear,0.21593924706774323,0.005964904040270763,False,0.4620156444260257,0.10776907807676153,1.1803797468354396
running_xor,Qwen2.5-7B-Instruct,16,,5,-29.875118833617563,-30.656243008749364,-29.842165441453446,-31.013851704151143,-14.099833313905268,-14.880957489037067,linear,quadratic,,,False,0.35317688378033213,-0.12500000000000017,1.326249995259948
running_xor,Qwen2.5-7B-Instruct,32,,6,-33.088297043148636,-33.50477810469253,-31.491423367353434,-32.11614495966927,-19.05389844358303,-19.47037950512692,linear,linear,0.1563419189503268,0.0035369504882910765,False,0.36908701806416816,0.010330334448520976,1.207465537908301
running_xor,Qwen2.5-7B-Instruct,64,,6,-39.12068968564028,-39.53717074718417,-37.194395658788956,-37.81911725110479,-16.05525223651994,-16.471733298063832,linear,linear,0.03801742920098479,0.0013482731608764125,False,-2.5719955012853553,,
running_sum_local,Mistral-7B-Instruct-v0.3,16,2.0,5,-33.32282170258239,-34.10394587771419,-35.07341490832806,-36.24510117102576,-11.065594382382113,-11.846718557513912,quadratic,quadratic,,,False,0.42382342533616413,0.07187500000000004,1.2321428571428594
running_sum_local,Mistral-7B-Instruct-v0.3,16,4.0,5,-31.80953752558419,-32.59066170071599,-29.87546730215753,-31.04715356485523,-18.471123541953425,-19.252247717085226,linear,linear,,,False,0.5885617760617758,0.12701612903225806,0.9166666666666661
running_sum_local,Mistral-7B-Instruct-v0.3,16,8.0,5,-37.341309859081264,-38.122434034213065,-36.812086359183596,-37.98377262188129,-23.345383815299257,-24.126507990431058,linear,linear,,,False,0.6018835616438356,0.2867647058823529,1.0833333333333344
running_sum_local,Mistral-7B-Instruct-v0.3,32,2.0,6,-29.673848036775738,-30.09032909831963,-33.680773444807414,-34.30549503712325,-15.417257835842861,-15.833738897386752,quadratic,quadratic,0.11306622945461438,0.02144003107796802,False,0.5287948604465738,-0.007142857142857443,0.6097715736040609
running_sum_local,Mistral-7B-Instruct-v0.3,32,4.0,6,-40.73087525186622,-41.147356313410114,-45.924217639395536,-46.54893923171137,-23.712021957631602,-24.128503019175493,quadratic,quadratic,0.01166985449926529,0.003471283479077604,False,0.5711753731343283,-0.04208672033147821,0.6293172479332422
running_sum_local,Mistral-7B-Instruct-v0.3,32,8.0,6,-52.72161141040286,-53.13809247194675,-55.950670269287954,-56.57539186160379,-29.453691044348233,-29.870172105892124,quadratic,quadratic,0.0006403666211649052,0.0003885524134138357,False,0.6378152475705694,-0.11143555099725735,0.7158721383975034
running_sum_local,Mistral-7B-Instruct-v0.3,64,2.0,6,-31.39109786699646,-31.80757892854035,-36.824626646440464,-37.4493482387563,-15.544994783247567,-15.961475844791458,quadratic,quadratic,0.02285993325072343,0.012284773295672943,False,0.5229957922588182,0.022612345555998385,0.5668132643108689
running_sum_local,Mistral-7B-Instruct-v0.3,64,4.0,6,-43.626494691246144,-44.042975752790035,-44.63718704438601,-45.26190863670185,-25.459342633500967,-25.875823695044858,quadratic,quadratic,0.0010735909068648366,0.001323220302783354,True,0.7374518554332601,0.09400788135174795,0.9231717231740216
running_sum_local,Mistral-7B-Instruct-v0.3,64,8.0,6,-52.66015751682038,-53.07663857836427,-50.956550635869334,-51.58127222818517,-31.227285264965232,-31.643766326509123,linear,linear,0.0015665870174624494,0.00014174241063904356,False,1.3771666333466681,-0.011027954004760743,1.3771666333466905
running_sum_local,Phi-3.5-mini-instruct,16,2.0,5,-23.766300103364596,-24.547424278496397,-28.5866161938985,-29.758302456596198,-9.322958880784974,-10.104083055916773,quadratic,quadratic,,,False,0.3759509624197983,0.05681818181818204,0.5357142857142857
running_sum_local,Phi-3.5-mini-instruct,16,4.0,5,-26.8950925957105,-27.676216770842302,-30.547762664396444,-31.71944892709414,-15.319793229715518,-16.10091740484732,quadratic,quadratic,,,False,0.29883377135348205,-0.42038938888082217,0.5151869158878505
running_sum_local,Phi-3.5-mini-instruct,16,8.0,5,-27.40459511335964,-28.18571928849144,-32.74456249539969,-33.91624875809739,-18.34212674849872,-19.123250923630522,quadratic,quadratic,,,False,0.24610894941634245,-0.2170593976575573,0.4565217391304348
running_sum_local,Phi-3.5-mini-instruct,32,2.0,6,-43.7568519484776,-44.173333010021494,-47.28698554899161,-47.911707141307446,-14.81490769459823,-15.23138875614212,quadratic,quadratic,0.0033928900501996304,0.001799688214074521,False,0.28313267112900187,0.04384439013168452,0.7188776115366654
running_sum_local,Phi-3.5-mini-instruct,32,4.0,6,-35.35031296211712,-35.766794023661014,-48.08664370883454,-48.71136530115038,-22.03183509029656,-22.44831615184045,quadratic,quadratic,0.001967596290830094,0.00899931604620466,True,0.4717616299253601,0.08162313432835831,1.1989620337477789
running_sum_local,Phi-3.5-mini-instruct,32,8.0,6,-47.3114186252034,-47.72789968674729,-52.618514412028944,-53.24323600434478,-29.03420045404583,-29.45068151558972,quadratic,quadratic,0.0035775490810987473,0.0011551445561564372,False,0.2631229916385291,-0.204913843487803,0.6707848837209301
running_sum_local,Phi-3.5-mini-instruct,64,2.0,6,-43.87109157084346,-44.28757263238735,-51.26533854790838,-51.890060140224215,-14.491134900377858,-14.90761596192175,quadratic,quadratic,0.00018360271841112038,0.001868060068605795,True,0.6047055892958816,0.11476293103448297,0.8605689368770718
running_sum_local,Phi-3.5-mini-instruct,64,4.0,6,-49.2144349284437,-49.63091598998759,-49.8490103227938,-50.47373191510964,-23.56905779403986,-23.98553885558375,quadratic,quadratic,0.0002385598005574912,0.0005114047645183741,True,0.21707045108448447,-0.3788270779692402,0.9834558823529405
running_sum_local,Phi-3.5-mini-instruct,64,8.0,6,-46.32818745662294,-46.74466851816683,-45.92589414764057,-46.550615739956406,-31.50727043297052,-31.923751494514413,linear,linear,0.007512037440654362,0.0008074930074901133,False,0.2830173521850885,-0.3194444444444423,0.6278964093026029
running_sum_local,Qwen2.5-14B-Instruct,16,2.0,5,-26.27886963916778,-27.05999381429958,-26.157835185455653,-27.32952144815335,-6.941341720001436,-7.722465895133235,linear,quadratic,,,False,0.5579473206176203,0.13322368421052633,0.7250000000000001
running_sum_local,Qwen2.5-14B-Instruct,16,4.0,5,-26.871704726489803,-27.652828901621604,-25.078312588640063,-26.249998851337764,-10.668466056576424,-11.449590231708223,linear,linear,,,False,0.42446380697050884,-0.3133031682816534,0.6710526315789473
running_sum_local,Qwen2.5-14B-Instruct,16,8.0,5,-28.6735977217186,-29.4547218968504,-26.674036305634573,-27.84572256833227,-14.160566314262173,-14.941690489393972,linear,linear,,,False,15.150431034481405,,
running_sum_local,Qwen2.5-14B-Instruct,32,2.0,6,-40.47333240380814,-40.889813465352034,-39.72420810541809,-40.34892969773393,-12.438353705905392,-12.854834767449283,linear,linear,0.03875367187325233,0.0012664761845530227,False,0.5909575637642922,0.16282894736842177,0.9797297297297299
running_sum_local,Qwen2.5-14B-Instruct,32,4.0,6,-31.837713742368564,-32.254194803912455,-29.877603679873275,-30.50232527218911,-18.047247843328666,-18.463728904872557,linear,linear,0.2250599022427764,0.004442302441444151,False,1.5103437618866487,,
running_sum_local,Qwen2.5-14B-Instruct,32,8.0,6,-31.82313083482081,-32.2396118963647,-30.20357127468023,-30.828292866996065,-22.832642499690547,-23.249123561234438,linear,linear,0.1805638541713457,0.0043501718294321895,False,0.6958055208983156,-0.257228304339126,0.8929153647196038
running_sum_local,Qwen2.5-14B-Instruct,64,2.0,6,-35.96901540138589,-36.38549646292978,-36.506971410464914,-37.13169300278075,-13.104344747258796,-13.520825808802687,quadratic,quadratic,0.06192004700509853,0.003431128781204941,False,0.2964112363311759,-0.05311137684517964,1.3841283787228775
running_sum_local,Qwen2.5-14B-Instruct,64,4.0,6,-34.151311012971185,-34.567792074515076,-33.095929151948866,-33.7206507442647,-20.49536965227292,-20.91185071381681,linear,linear,0.13725985201683424,0.0033093964740584635,False,0.42010759960649324,0.2019924006235386,1.2010795254057889
running_sum_local,Qwen2.5-14B-Instruct,64,8.0,6,-31.251308037092713,-31.667789098636604,-29.541651593089483,-30.16637318540532,-25.920132035707507,-26.336613097251398,linear,linear,0.23688726978384558,0.004952821843674027,False,0.49681267806267815,-0.11720385731259869,0.8387473868725636
running_sum_local,Qwen2.5-32B-Instruct,16,2.0,5,-28.369325065441707,-29.150449240573508,-31.88267129018655,-33.05435755288425,-6.361230415720353,-7.142354590852152,quadratic,quadratic,,,False,0.340403411268685,-0.4374999999999953,0.533108108108108
running_sum_local,Qwen2.5-32B-Instruct,16,4.0,5,-36.83317162357715,-37.61429579870895,-39.90195740974324,-41.07364367244094,-10.482822470255314,-11.263946645387113,quadratic,quadratic,,,False,0.8301806953339435,0.3250000000000034,1.294207317073175
running_sum_local,Qwen2.5-32B-Instruct,16,8.0,5,-26.807280903914656,-27.588405079046456,-31.930084514757482,-33.10177077745518,-10.940163277650313,-11.721287452782112,quadratic,quadratic,,,False,0.41183886838868383,0.008208924461845376,0.737499999999999
running_sum_local,Qwen2.5-32B-Instruct,32,2.0,6,-40.43972452267635,-40.85620558422024,-38.626649985413344,-39.25137157772918,-11.327673913295289,-11.74415497483918,linear,linear,0.0528518334270724,0.0010173496053211722,False,2.014550144822593,,
running_sum_local,Qwen2.5-32B-Instruct,32,4.0,6,-40.10419106187648,-40.52067212342037,-38.48291846439962,-39.107640056715454,-16.593862781572916,-17.010343843116807,linear,linear,0.05190709759149931,0.0010269000780096101,False,1.4839706432311168,-0.4400317693970314,1.4100811014711467
running_sum_local,Qwen2.5-32B-Instruct,32,8.0,6,-39.29186930523065,-39.70835036677454,-37.30100827278685,-37.92572986510269,-20.032152294177294,-20.448633355721185,linear,linear,0.06039948077333674,0.0015839933727595963,False,-0.2698762924667696,-0.1469126799421451,1.2348875105130337
running_sum_local,Qwen2.5-32B-Instruct,64,2.0,6,-29.90454839066328,-30.32102945220717,-29.654698056184458,-30.279419648500294,-13.028242002335379,-13.44472306387927,linear,linear,0.17961799202588444,0.008545297439879522,False,0.4880261141240273,0.16381835937500003,1.1012955580865569
running_sum_local,Qwen2.5-32B-Instruct,64,4.0,6,-31.34781869103164,-31.764299752575532,-31.884657065980036,-32.50937865829587,-20.211582878538536,-20.628063940082427,quadratic,quadratic,0.12194166434178526,0.007683334255896249,False,0.5125261026904379,-0.2287121650231681,0.8763586956521744
running_sum_local,Qwen2.5-32B-Instruct,64,8.0,6,-31.333505464700934,-31.749986526244825,-30.918723907432536,-31.543445499748373,-24.20380693963932,-24.620288001183212,linear,linear,0.1449595289524029,0.0065584934949978356,False,0.4504690388170057,0.1730701133144476,1.0751993166287026
running_sum_local,Qwen2.5-7B-Instruct,16,2.0,5,-31.61081646368803,-32.39194063881983,-33.34912107872891,-34.52080734142661,-13.12964711837494,-13.91077129350674,quadratic,quadratic,,,False,0.5690074626865671,-0.34103260869565266,0.6372377622377622
running_sum_local,Qwen2.5-7B-Instruct,16,4.0,5,-30.21827772141401,-30.999401896545812,-31.402479347594706,-32.574165610292404,-18.016780063993018,-18.79790423912482,quadratic,quadratic,,,False,0.4775878638110928,0.14700704225352115,1.0624999999999978
running_sum_local,Qwen2.5-7B-Instruct,16,8.0,5,-30.393448100802416,-31.174572275934217,-33.58998658852223,-34.76167285121993,-19.147997330715082,-19.929121505846883,quadratic,quadratic,,,False,0.4579437022900764,-0.13815789473684306,0.6371268656716418
running_sum_local,Qwen2.5-7B-Instruct,32,2.0,6,-34.60722848919836,-35.023709550742254,-32.692339157747995,-33.31706075006383,-17.779065795283408,-18.1955468568273,linear,linear,0.01282644214975516,0.0028692952036799917,False,0.5333080019717388,0.09418064709406833,1.0598334096220121
running_sum_local,Qwen2.5-7B-Instruct,32,4.0,6,-49.94018006856127,-50.35666113010516,-48.869104417726106,-49.49382601004194,-25.486613585858116,-25.903094647402007,linear,linear,0.009231882745353163,0.00024339347144121433,False,0.7411446784922381,-0.007342076751560595,1.2421717761014333
running_sum_local,Qwen2.5-7B-Instruct,32,8.0,6,-40.33057649252282,-40.747057554066714,-39.51353932944437,-40.13826092176021,-30.913821853270335,-31.330302914814226,linear,linear,0.0060788304344780554,0.0018077876694857049,False,0.40270721925133607,-0.11518485915492589,0.551056338028169
running_sum_local,Qwen2.5-7B-Instruct,64,2.0,6,-42.34513841969044,-42.761619481234334,-40.37021908276848,-40.994940675084315,-17.661750820606798,-18.07823188215069,linear,linear,0.035743212365042284,0.0007808133445232953,False,-0.7418944188722529,,
running_sum_local,Qwen2.5-7B-Instruct,64,4.0,6,-46.5084769424178,-46.92495800396169,-45.18496567977118,-45.80968727208702,-26.340207409912246,-26.756688471456137,linear,linear,0.01747739071978336,0.0004135042170871029,False,0.5312922637256484,0.16902608530083785,0.8993182873610054
running_sum_local,Qwen2.5-7B-Instruct,64,8.0,6,-43.975245684665836,-44.39172674620973,-42.24303160834036,-42.867753200656196,-30.916801757371836,-31.333282818915727,linear,linear,0.0019752789662729094,0.0005849586677326188,False,0.7145935207823976,0.02574982193975495,1.0166719080517932
```

---

## Appendix F: Well-formedness tables

### Pooled per-task, per-model, per-method (TAR/SAR/VVR/CR/WFR/NEO/TAR|WF)
```csv
Task,Model,Method,N_trials,TAR,SAR,Vocab_Valid_Rate,Completion_Rate,Well_Formed_Rate,Normalized_Error_Onset,Overlong_Rate,Boundary_Fidelity,N_with_raw_output,TAR_given_WellFormed
cyclic_shift,Mistral-7B-Instruct-v0.3,Base,840,0.35600353422619047,0.034523809523809526,0.5785714285714286,0.9630952380952381,0.5595238095238095,0.13671409970238096,,,0,0.46317320478723406
cyclic_shift,Mistral-7B-Instruct-v0.3,Cellular_Bounds,3480,0.37202990301724137,0.05660919540229885,0.5695402298850575,0.8091954022988506,0.4362068965517241,0.14280374461206896,,,0,0.5153753911396575
cyclic_shift,Mistral-7B-Instruct-v0.3,All,4320,0.36891366464120373,0.052314814814814814,0.5712962962962963,0.8391203703703703,0.4601851851851852,0.14161964699074073,,,0,0.5030338279678068
cyclic_shift,Phi-3.5-mini-instruct,Base,420,0.3148344494047619,0.04285714285714286,0.7976190476190477,0.5547619047619048,0.4523809523809524,0.12523251488095238,,,0,0.38924753289473685
cyclic_shift,Phi-3.5-mini-instruct,Cellular_Bounds,1740,0.3468121408045977,0.040229885057471264,0.5804597701149425,0.6229885057471264,0.3660919540229885,0.11867591594827587,,,0,0.46915472919937207
cyclic_shift,Phi-3.5-mini-instruct,All,2160,0.34059425636574076,0.040740740740740744,0.6226851851851852,0.6097222222222223,0.38287037037037036,0.11995081018518519,,,0,0.4507963648730351
cyclic_shift,Qwen2.5-14B-Instruct,Base,420,0.4647135416666667,0.07142857142857142,0.43333333333333335,0.9738095238095238,0.4095238095238095,0.1919921875,,,0,0.6590661337209303
cyclic_shift,Qwen2.5-14B-Instruct,Cellular_Bounds,1740,0.6270833333333333,0.19712643678160918,0.49310344827586206,0.7804597701149425,0.37471264367816093,0.30725799209770116,,,0,0.8458049750766872
cyclic_shift,Qwen2.5-14B-Instruct,All,2160,0.5955114293981482,0.1726851851851852,0.48148148148148145,0.8180555555555555,0.3814814814814815,0.28484519675925923,,,0,0.8068255081917476
cyclic_shift,Qwen2.5-32B-Instruct,Base,420,0.5110770089285714,0.10952380952380952,0.5357142857142857,0.919047619047619,0.5214285714285715,0.2384858630952381,,,0,0.6857341609589042
cyclic_shift,Qwen2.5-32B-Instruct,Cellular_Bounds,1740,0.7014861709770115,0.2540229885057471,0.5683908045977012,0.8057471264367816,0.4442528735632184,0.38133081896551724,,,0,0.8797602684346701
cyclic_shift,Qwen2.5-32B-Instruct,All,2160,0.6644621672453703,0.22592592592592592,0.562037037037037,0.8277777777777777,0.45925925925925926,0.3535554108796296,,,0,0.8369258757560484
cyclic_shift,Qwen2.5-7B-Instruct,Base,420,0.42253534226190476,0.05476190476190476,0.4595238095238095,0.9095238095238095,0.40714285714285714,0.1630487351190476,,,0,0.6188550804093568
cyclic_shift,Qwen2.5-7B-Instruct,Cellular_Bounds,1740,0.5203573994252874,0.15057471264367817,0.4839080459770115,0.6511494252873563,0.30977011494252876,0.2526782507183908,,,0,0.8189717648423006
cyclic_shift,Qwen2.5-7B-Instruct,All,2160,0.5013364438657407,0.13194444444444445,0.4791666666666667,0.7013888888888888,0.3287037037037037,0.23525028935185185,,,0,0.770774647887324
character_substitution,Mistral-7B-Instruct-v0.3,Base,600,0.39130208333333333,0.04833333333333333,0.975,0.8933333333333333,0.8716666666666667,0.170078125,,,0,0.41742351816443596
character_substitution,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2280,0.44702576754385964,0.09649122807017543,0.9565789473684211,0.8473684210526315,0.8087719298245614,0.2230811403508772,,,0,0.47859612255965295
character_substitution,Mistral-7B-Instruct-v0.3,All,2880,0.4354166666666667,0.08645833333333333,0.9604166666666667,0.8569444444444444,0.821875,0.2120388454861111,,,0,0.46507974228981835
character_substitution,Phi-3.5-mini-instruct,Base,300,0.306875,0.04666666666666667,0.9966666666666667,0.6133333333333333,0.6133333333333333,0.11213541666666667,,,0,0.3589504076086957
character_substitution,Phi-3.5-mini-instruct,Cellular_Bounds,1140,0.31684484649122807,0.05526315789473684,1.0,0.7394736842105263,0.7394736842105263,0.11800986842105263,,,0,0.34252669039145905
character_substitution,Phi-3.5-mini-instruct,All,1440,0.31476779513888886,0.05347222222222222,0.9993055555555556,0.7131944444444445,0.7131944444444445,0.11678602430555556,,,0,0.34546920642648493
character_substitution,Qwen2.5-14B-Instruct,Base,600,0.45484375,0.07166666666666667,0.96,0.9533333333333334,0.92,0.22106770833333333,,,0,0.4715523097826087
character_substitution,Qwen2.5-14B-Instruct,Cellular_Bounds,2280,0.6383018092105263,0.2706140350877193,0.9802631578947368,0.9118421052631579,0.8934210526315789,0.39601151315789473,,,0,0.6490243004418262
character_substitution,Qwen2.5-14B-Instruct,All,2880,0.6000813802083333,0.22916666666666666,0.9760416666666667,0.9204861111111111,0.8989583333333333,0.3595648871527778,,,0,0.6111855446118193
character_substitution,Qwen2.5-32B-Instruct,Base,600,0.48171875,0.095,0.9616666666666667,0.8666666666666667,0.84,0.255546875,,,0,0.5207403273809523
character_substitution,Qwen2.5-32B-Instruct,Cellular_Bounds,2280,0.7136307565789474,0.3320175438596491,0.9714912280701754,0.9276315789473685,0.9013157894736842,0.44944490131578946,,,0,0.7250988442822385
character_substitution,Qwen2.5-32B-Instruct,All,2880,0.6653157552083333,0.2826388888888889,0.9694444444444444,0.9149305555555556,0.8885416666666667,0.4090494791666667,,,0,0.6848500390777648
character_substitution,Qwen2.5-7B-Instruct,Base,600,0.4202604166666667,0.06166666666666667,0.91,0.9083333333333333,0.8266666666666667,0.19244791666666666,,,0,0.4529674899193548
character_substitution,Qwen2.5-7B-Instruct,Cellular_Bounds,2280,0.5358072916666666,0.18771929824561404,0.874561403508772,0.8065789473684211,0.7087719298245614,0.30102110745614036,,,0,0.6134166150990099
character_substitution,Qwen2.5-7B-Instruct,All,2880,0.5117350260416667,0.16145833333333334,0.8819444444444444,0.8277777777777777,0.7333333333333333,0.2784016927083333,,,0,0.575735381155303
random_permutation,Mistral-7B-Instruct-v0.3,Base,600,0.3792447916666667,0.04,0.995,0.9083333333333333,0.9033333333333333,0.15033854166666666,,,0,0.3963618542435424
random_permutation,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2280,0.4231359649122807,0.07763157894736843,0.9780701754385965,0.8333333333333334,0.8153508771929825,0.18380619517543859,,,0,0.449014927380312
random_permutation,Mistral-7B-Instruct-v0.3,All,2880,0.41399197048611114,0.06979166666666667,0.9815972222222222,0.8489583333333334,0.8336805555555555,0.17683376736111112,,,0,0.4371290608079967
random_permutation,Phi-3.5-mini-instruct,Base,300,0.324375,0.04666666666666667,1.0,0.5933333333333334,0.5933333333333334,0.11421875,,,0,0.39115168539325845
random_permutation,Phi-3.5-mini-instruct,Cellular_Bounds,1140,0.3417626096491228,0.054385964912280704,1.0,0.7333333333333333,0.7333333333333333,0.11437774122807018,,,0,0.3769437799043062
random_permutation,Phi-3.5-mini-instruct,All,1440,0.3381401909722222,0.05277777777777778,1.0,0.7041666666666667,0.7041666666666667,0.11434461805555556,,,0,0.37943786982248523
random_permutation,Qwen2.5-14B-Instruct,Base,600,0.44494791666666667,0.07,0.9883333333333333,0.9733333333333334,0.9616666666666667,0.19953125,,,0,0.4497671143847487
random_permutation,Qwen2.5-14B-Instruct,Cellular_Bounds,2280,0.6243352521929825,0.24298245614035088,0.9916666666666667,0.9100877192982456,0.9026315789473685,0.3511444627192982,,,0,0.6326454689018465
random_permutation,Qwen2.5-14B-Instruct,All,2880,0.586962890625,0.20694444444444443,0.9909722222222223,0.9232638888888889,0.9149305555555556,0.3195583767361111,,,0,0.5925996204933587
random_permutation,Qwen2.5-32B-Instruct,Base,600,0.4991145833333333,0.095,0.995,0.8583333333333333,0.855,0.25091145833333334,,,0,0.5328338206627681
random_permutation,Qwen2.5-32B-Instruct,Cellular_Bounds,2280,0.720141173245614,0.3267543859649123,0.9942982456140351,0.9228070175438596,0.9179824561403509,0.4449424342105263,,,0,0.7248491997133302
random_permutation,Qwen2.5-32B-Instruct,All,2880,0.6740939670138889,0.27847222222222223,0.9944444444444445,0.909375,0.9048611111111111,0.4045193142361111,,,0,0.6870503165771297
random_permutation,Qwen2.5-7B-Instruct,Base,600,0.4170052083333333,0.056666666666666664,0.8966666666666666,0.88,0.7833333333333333,0.18622395833333333,,,0,0.4606715425531915
random_permutation,Qwen2.5-7B-Instruct,Cellular_Bounds,2280,0.5278919956140351,0.16535087719298244,0.8394736842105263,0.7973684210526316,0.6728070175438596,0.281640625,,,0,0.6140808344198174
random_permutation,Qwen2.5-7B-Instruct,All,2880,0.5047905815972222,0.14270833333333333,0.8513888888888889,0.8145833333333333,0.6958333333333333,0.26176215277777776,,,0,0.5781016092814372
conditional_shift,Mistral-7B-Instruct-v0.3,Base,600,0.3603385416666667,0.03,0.73,0.9683333333333334,0.7066666666666667,0.1446875,,,0,0.39047759433962265
conditional_shift,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2280,0.39373629385964914,0.05043859649122807,0.7289473684210527,0.8565789473684211,0.6070175438596491,0.16882538377192982,,,0,0.4468027456647399
conditional_shift,Mistral-7B-Instruct-v0.3,All,2880,0.38677842881944446,0.04618055555555556,0.7291666666666666,0.8798611111111111,0.6277777777777778,0.16379665798611112,,,0,0.43359375
conditional_shift,Phi-3.5-mini-instruct,Base,300,0.34734375,0.03333333333333333,0.8133333333333334,0.55,0.44,0.13979166666666668,,,0,0.4322916666666667
conditional_shift,Phi-3.5-mini-instruct,Cellular_Bounds,1140,0.3658168859649123,0.03245614035087719,0.65,0.6622807017543859,0.4456140350877193,0.10875822368421052,,,0,0.4555548720472441
conditional_shift,Phi-3.5-mini-instruct,All,1440,0.36196831597222223,0.03263888888888889,0.6840277777777778,0.6388888888888888,0.4444444444444444,0.11522352430555556,,,0,0.4507568359375
conditional_shift,Qwen2.5-14B-Instruct,Base,300,0.42640625,0.056666666666666664,0.68,0.9133333333333333,0.6333333333333333,0.16895833333333332,,,0,0.4993421052631579
conditional_shift,Qwen2.5-14B-Instruct,Cellular_Bounds,1140,0.5459429824561404,0.1324561403508772,0.6921052631578948,0.8640350877192983,0.6035087719298246,0.2216282894736842,,,0,0.6369685683139535
conditional_shift,Qwen2.5-14B-Instruct,All,1440,0.5210394965277778,0.11666666666666667,0.6895833333333333,0.8743055555555556,0.6097222222222223,0.21065538194444444,,,0,0.607186076309795
conditional_shift,Qwen2.5-32B-Instruct,Base,300,0.51375,0.08,0.9166666666666666,0.9366666666666666,0.8633333333333333,0.22552083333333334,,,0,0.5459700772200772
conditional_shift,Qwen2.5-32B-Instruct,Cellular_Bounds,1140,0.6705592105263158,0.2,0.8342105263157895,0.9078947368421053,0.7517543859649123,0.31668037280701755,,,0,0.7167626896149358
conditional_shift,Qwen2.5-32B-Instruct,All,1440,0.637890625,0.175,0.8513888888888889,0.9138888888888889,0.775,0.2976888020833333,,,0,0.6771253360215054
conditional_shift,Qwen2.5-7B-Instruct,Base,600,0.3622916666666667,0.05333333333333334,0.5433333333333333,0.8633333333333333,0.4816666666666667,0.12278645833333333,,,0,0.4726427335640138
conditional_shift,Qwen2.5-7B-Instruct,Cellular_Bounds,2280,0.3838815789473684,0.05043859649122807,0.33245614035087717,0.8285087719298245,0.2899122807017544,0.11086896929824562,,,0,0.5630200453857791
conditional_shift,Qwen2.5-7B-Instruct,All,2880,0.37938368055555555,0.051041666666666666,0.3763888888888889,0.8357638888888889,0.3298611111111111,0.11335177951388889,,,0,0.5355263157894737
composed_permutation,Mistral-7B-Instruct-v0.3,Base,400,0.1794140625,0.0,0.995,0.91,0.905,0.02046875,,,0,0.18637776243093923
composed_permutation,Mistral-7B-Instruct-v0.3,Cellular_Bounds,1520,0.1466282894736842,0.0006578947368421052,0.9822368421052632,0.8644736842105263,0.8486842105263158,0.01797902960526316,,,0,0.1491763565891473
composed_permutation,Mistral-7B-Instruct-v0.3,All,1920,0.15345865885416668,0.0005208333333333333,0.9848958333333333,0.8739583333333333,0.8604166666666667,0.018497721354166666,,,0,0.15732823849878935
composed_permutation,Phi-3.5-mini-instruct,Base,200,0.15203125,0.0,1.0,0.615,0.615,0.011953125,,,0,0.16323678861788618
composed_permutation,Phi-3.5-mini-instruct,Cellular_Bounds,760,0.1571546052631579,0.0,1.0,0.6842105263157895,0.6842105263157895,0.01722861842105263,,,0,0.16712740384615385
composed_permutation,Phi-3.5-mini-instruct,All,960,0.15608723958333334,0.0,1.0,0.6697916666666667,0.6697916666666667,0.016129557291666666,,,0,0.16638316485225504
composed_permutation,Qwen2.5-14B-Instruct,Base,400,0.1931640625,0.0,0.9975,0.97,0.9675,0.0251171875,,,0,0.19597868217054262
composed_permutation,Qwen2.5-14B-Instruct,Cellular_Bounds,1520,0.1819078947368421,0.002631578947368421,0.9986842105263158,0.9092105263157895,0.9078947368421053,0.01850328947368421,,,0,0.1868546195652174
composed_permutation,Qwen2.5-14B-Instruct,All,1920,0.1842529296875,0.0020833333333333333,0.9984375,0.921875,0.9203125,0.019881184895833334,,,0,0.18885292869269948
composed_permutation,Qwen2.5-32B-Instruct,Base,400,0.16171875,0.0,1.0,0.86,0.86,0.0091015625,,,0,0.16783248546511628
composed_permutation,Qwen2.5-32B-Instruct,Cellular_Bounds,1520,0.13092105263157894,0.0,1.0,0.9131578947368421,0.9131578947368421,0.00600328947368421,,,0,0.13236221181556196
composed_permutation,Qwen2.5-32B-Instruct,All,1920,0.13733723958333333,0.0,1.0,0.9020833333333333,0.9020833333333333,0.0066487630208333336,,,0,0.13940711605080833
composed_permutation,Qwen2.5-7B-Instruct,Base,400,0.1569921875,0.0,0.96,0.9,0.87,0.009140625,,,0,0.16410739942528735
composed_permutation,Qwen2.5-7B-Instruct,Cellular_Bounds,1520,0.10867598684210526,0.0,0.8276315789473684,0.8230263157894737,0.6888157894736842,0.0021073190789473685,,,0,0.11294173829990449
composed_permutation,Qwen2.5-7B-Instruct,All,1920,0.11874186197916667,0.0,0.8552083333333333,0.8390625,0.7265625,0.0035725911458333332,,,0,0.12570564516129032
copy_previous,Mistral-7B-Instruct-v0.3,Base,200,0.57015625,0.04,1.0,0.93,0.93,0.11046875,,,0,0.5824092741935484
copy_previous,Mistral-7B-Instruct-v0.3,Cellular_Bounds,760,0.5533511513157895,0.018421052631578946,0.9986842105263158,0.8921052631578947,0.8907894736842106,0.1098889802631579,,,0,0.5715472673559823
copy_previous,Mistral-7B-Instruct-v0.3,All,960,0.5568522135416667,0.022916666666666665,0.9989583333333333,0.9,0.8989583333333333,0.110009765625,,,0,0.573888325608343
copy_previous,Qwen2.5-14B-Instruct,Base,200,0.861171875,0.59,1.0,0.915,0.915,0.721796875,,,0,0.8845628415300546
copy_previous,Qwen2.5-14B-Instruct,Cellular_Bounds,760,0.7399671052631579,0.23684210526315788,0.9986842105263158,0.7842105263157895,0.7828947368421053,0.43209292763157897,,,0,0.7713760504201681
copy_previous,Qwen2.5-14B-Instruct,All,960,0.7652180989583334,0.3104166666666667,0.9989583333333333,0.8114583333333333,0.8104166666666667,0.49244791666666665,,,0,0.7979996786632391
copy_previous,Qwen2.5-32B-Instruct,Base,200,0.859453125,0.495,1.0,0.955,0.955,0.689296875,,,0,0.8704188481675392
copy_previous,Qwen2.5-32B-Instruct,Cellular_Bounds,760,0.8167146381578947,0.27631578947368424,1.0,0.8789473684210526,0.8789473684210526,0.48955592105263157,,,0,0.8366158308383234
copy_previous,Qwen2.5-32B-Instruct,All,960,0.8256184895833333,0.321875,1.0,0.8947916666666667,0.8947916666666667,0.5311686197916666,,,0,0.8441319848661234
copy_previous,Qwen2.5-7B-Instruct,Base,200,0.528984375,0.06,0.505,0.885,0.445,0.1675,,,0,0.6530898876404494
copy_previous,Qwen2.5-7B-Instruct,Cellular_Bounds,760,0.4169819078947368,0.010526315789473684,0.25263157894736843,0.8171052631578948,0.20921052631578949,0.10213815789473685,,,0,0.5678066037735849
copy_previous,Qwen2.5-7B-Instruct,All,960,0.4403157552083333,0.020833333333333332,0.30520833333333336,0.83125,0.25833333333333336,0.11575520833333333,,,0,0.5984122983870968
running_max,Mistral-7B-Instruct-v0.3,Base,600,0.57125,0.085,0.9983333333333333,0.6116666666666667,0.61,0.20182291666666666,,,0,0.5890966530054644
running_max,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2280,0.4676466557017544,0.03464912280701754,0.9969298245614036,0.41535087719298247,0.41271929824561404,0.12820723684210528,,,0,0.5881874335812965
running_max,Mistral-7B-Instruct-v0.3,All,2880,0.4892306857638889,0.04513888888888889,0.9972222222222222,0.45625,0.45381944444444444,0.14354383680555555,,,0,0.5884420428462127
running_max,Qwen2.5-14B-Instruct,Base,600,0.9219010416666666,0.735,1.0,0.9933333333333333,0.9933333333333333,0.810390625,,,0,0.9243655620805369
running_max,Qwen2.5-14B-Instruct,Cellular_Bounds,2280,0.8487733004385964,0.4381578947368421,0.999561403508772,0.8679824561403509,0.8675438596491228,0.6105263157894737,,,0,0.8701892694641051
running_max,Qwen2.5-14B-Instruct,All,2880,0.8640082465277777,0.5,0.9996527777777777,0.8940972222222222,0.89375,0.6521647135416667,,,0,0.8827335858585859
running_max,Qwen2.5-32B-Instruct,Base,600,0.9790364583333333,0.865,1.0,0.98,0.98,0.910390625,,,0,0.9825414540816326
running_max,Qwen2.5-32B-Instruct,Cellular_Bounds,2280,0.9004248903508771,0.5565789473684211,1.0,0.9377192982456141,0.9377192982456141,0.7123492324561403,,,0,0.9014557998129092
running_max,Qwen2.5-32B-Instruct,All,2880,0.9168023003472222,0.6208333333333333,1.0,0.9465277777777777,0.9465277777777777,0.7536078559027778,,,0,0.9189460289801907
running_max,Qwen2.5-7B-Instruct,Base,600,0.6971614583333333,0.22166666666666668,0.7033333333333334,0.8666666666666667,0.5983333333333334,0.307578125,,,0,0.7730675487465181
running_max,Qwen2.5-7B-Instruct,Cellular_Bounds,2280,0.6579632675438597,0.1736842105263158,0.7039473684210527,0.7592105263157894,0.5241228070175439,0.27884457236842103,,,0,0.7754314853556485
running_max,Qwen2.5-7B-Instruct,All,2880,0.6661295572916667,0.18368055555555557,0.7038194444444444,0.7815972222222223,0.5395833333333333,0.2848307291666667,,,0,0.7748853764478765
running_xor,Mistral-7B-Instruct-v0.3,Base,200,0.431328125,0.0,0.775,0.83,0.61,0.0834375,,,0,0.4661885245901639
running_xor,Mistral-7B-Instruct-v0.3,Cellular_Bounds,760,0.46504934210526316,0.010526315789473684,0.8618421052631579,0.7144736842105263,0.6013157894736842,0.12117598684210526,,,0,0.5223263129102844
running_xor,Mistral-7B-Instruct-v0.3,All,960,0.45802408854166665,0.008333333333333333,0.84375,0.7385416666666667,0.603125,0.11331380208333333,,,0,0.5104976252158895
running_xor,Phi-3.5-mini-instruct,Base,100,0.57359375,0.03,1.0,0.92,0.92,0.2675,,,0,0.576766304347826
running_xor,Phi-3.5-mini-instruct,Cellular_Bounds,380,0.5894736842105263,0.034210526315789476,1.0,0.8447368421052631,0.8447368421052631,0.20386513157894737,,,0,0.6104458722741433
running_xor,Phi-3.5-mini-instruct,All,480,0.5861653645833333,0.03333333333333333,1.0,0.8604166666666667,0.8604166666666667,0.21712239583333334,,,0,0.602943401937046
running_xor,Qwen2.5-14B-Instruct,Base,200,0.74828125,0.18,1.0,0.895,0.895,0.458828125,,,0,0.7692912011173184
running_xor,Qwen2.5-14B-Instruct,Cellular_Bounds,760,0.7156455592105263,0.06578947368421052,0.9973684210526316,0.925,0.9223684210526316,0.30941611842105265,,,0,0.7292038159771754
running_xor,Qwen2.5-14B-Instruct,All,960,0.7224446614583333,0.08958333333333333,0.9979166666666667,0.91875,0.9166666666666666,0.34054361979166664,,,0,0.7373579545454545
running_xor,Qwen2.5-32B-Instruct,Base,200,0.781015625,0.24,1.0,0.94,0.94,0.49453125,,,0,0.7955452127659575
running_xor,Qwen2.5-32B-Instruct,Cellular_Bounds,760,0.7956208881578948,0.11315789473684211,1.0,0.9302631578947368,0.9302631578947368,0.34954769736842106,,,0,0.8028200141442716
running_xor,Qwen2.5-32B-Instruct,All,960,0.792578125,0.13958333333333334,1.0,0.9322916666666666,0.9322916666666666,0.3797526041666667,,,0,0.8012918994413408
running_xor,Qwen2.5-7B-Instruct,Base,200,0.410546875,0.01,0.88,0.585,0.5,0.085234375,,,0,0.46171875
running_xor,Qwen2.5-7B-Instruct,Cellular_Bounds,760,0.4013569078947368,0.006578947368421052,0.5236842105263158,0.6618421052631579,0.3631578947368421,0.07008634868421053,,,0,0.5159080615942029
running_xor,Qwen2.5-7B-Instruct,All,960,0.403271484375,0.007291666666666667,0.5979166666666667,0.6458333333333334,0.39166666666666666,0.0732421875,,,0,0.5014960106382979
running_sum_local,Mistral-7B-Instruct-v0.3,Base,600,0.32114583333333335,0.005,0.7533333333333333,0.7866666666666666,0.5716666666666667,0.1225,,,0,0.38989613702623904
running_sum_local,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2280,0.29825246710526315,0.0017543859649122807,0.5280701754385965,0.6890350877192982,0.3478070175438597,0.10849095394736842,,,0,0.4160821248423707
running_sum_local,Mistral-7B-Instruct-v0.3,All,2880,0.30302191840277776,0.0024305555555555556,0.575,0.709375,0.39444444444444443,0.11140950520833333,,,0,0.4081756161971831
running_sum_local,Phi-3.5-mini-instruct,Base,300,0.38057291666666665,0.02,0.9033333333333333,0.91,0.8166666666666667,0.17010416666666667,,,0,0.4017857142857143
running_sum_local,Phi-3.5-mini-instruct,Cellular_Bounds,1140,0.367406798245614,0.021052631578947368,0.5385964912280702,0.7149122807017544,0.42894736842105263,0.13907620614035088,,,0,0.4888484151329243
running_sum_local,Phi-3.5-mini-instruct,All,1440,0.37014973958333336,0.020833333333333332,0.6145833333333334,0.7555555555555555,0.5097222222222222,0.14554036458333333,,,0,0.45978797683923706
running_sum_local,Qwen2.5-14B-Instruct,Base,600,0.54765625,0.11333333333333333,0.8016666666666666,0.78,0.665,0.34140625,,,0,0.6491619674185464
running_sum_local,Qwen2.5-14B-Instruct,Cellular_Bounds,2280,0.5486705043859649,0.05394736842105263,0.7087719298245614,0.8692982456140351,0.6403508771929824,0.24063870614035088,,,0,0.6431078767123287
running_sum_local,Qwen2.5-14B-Instruct,All,2880,0.5484592013888889,0.06631944444444444,0.728125,0.8506944444444444,0.6454861111111111,0.26163194444444443,,,0,0.6444072754168908
running_sum_local,Qwen2.5-32B-Instruct,Base,600,0.5901822916666667,0.11666666666666667,0.83,0.9,0.765,0.33731770833333335,,,0,0.6601307189542484
running_sum_local,Qwen2.5-32B-Instruct,Cellular_Bounds,2280,0.6012061403508772,0.04912280701754386,0.6991228070175438,0.9460526315789474,0.6754385964912281,0.24725877192982457,,,0,0.6869825487012987
running_sum_local,Qwen2.5-32B-Instruct,All,2880,0.5989095052083333,0.06319444444444444,0.7263888888888889,0.9364583333333333,0.6940972222222223,0.26602105034722223,,,0,0.6808169709854928
running_sum_local,Qwen2.5-7B-Instruct,Base,600,0.30919270833333334,0.0016666666666666668,0.8483333333333334,0.43,0.37166666666666665,0.095859375,,,0,0.40667040358744394
running_sum_local,Qwen2.5-7B-Instruct,Cellular_Bounds,2280,0.2789816337719298,0.002631578947368421,0.5890350877192982,0.581140350877193,0.3399122807017544,0.07155975877192983,,,0,0.34653225806451615
running_sum_local,Qwen2.5-7B-Instruct,All,2880,0.2852756076388889,0.0024305555555555556,0.6430555555555556,0.5496527777777778,0.34652777777777777,0.07662217881944444,,,0,0.3599699398797595
```

### By-alphabet-size / by-modulus breakdown
```csv
Task,Model,Method,Alphabet_Size,N_trials,TAR,SAR,Vocab_Valid_Rate,Completion_Rate,Well_Formed_Rate,Normalized_Error_Onset,TAR_given_WellFormed,Modulus
character_substitution,Mistral-7B-Instruct-v0.3,Base,2.0,200,0.588828125,0.105,0.995,0.965,0.96,0.260078125,0.593994140625,
character_substitution,Mistral-7B-Instruct-v0.3,Base,4.0,200,0.339140625,0.025,1.0,0.85,0.85,0.145,0.3588235294117647,
character_substitution,Mistral-7B-Instruct-v0.3,Base,8.0,200,0.2459375,0.015,0.93,0.865,0.805,0.10515625,0.2687305900621118,
character_substitution,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2.0,760,0.6339021381578948,0.1881578947368421,0.9789473684210527,0.8618421052631579,0.843421052631579,0.3395970394736842,0.656688767550702,
character_substitution,Mistral-7B-Instruct-v0.3,Cellular_Bounds,4.0,760,0.3810032894736842,0.056578947368421055,0.9736842105263158,0.8315789473684211,0.8078947368421052,0.17900904605263157,0.40515574104234525,
character_substitution,Mistral-7B-Instruct-v0.3,Cellular_Bounds,8.0,760,0.326171875,0.04473684210526316,0.9171052631578948,0.8486842105263158,0.775,0.15063733552631578,0.36133807300509335,
character_substitution,Phi-3.5-mini-instruct,Base,2.0,100,0.525,0.12,1.0,0.74,0.74,0.23234375,0.5494087837837838,
character_substitution,Phi-3.5-mini-instruct,Base,4.0,100,0.24109375,0.01,0.99,0.54,0.54,0.0653125,0.2902199074074074,
character_substitution,Phi-3.5-mini-instruct,Base,8.0,100,0.15453125,0.01,1.0,0.56,0.56,0.03875,0.17354910714285715,
character_substitution,Phi-3.5-mini-instruct,Cellular_Bounds,2.0,380,0.4821546052631579,0.11578947368421053,1.0,0.7763157894736842,0.7763157894736842,0.22051809210526316,0.5045021186440678,
character_substitution,Phi-3.5-mini-instruct,Cellular_Bounds,4.0,380,0.26069078947368424,0.018421052631578946,1.0,0.7078947368421052,0.7078947368421052,0.05875822368421053,0.2798559479553903,
character_substitution,Phi-3.5-mini-instruct,Cellular_Bounds,8.0,380,0.2076891447368421,0.031578947368421054,1.0,0.7342105263157894,0.7342105263157894,0.07475328947368422,0.23168682795698925,
character_substitution,Qwen2.5-14B-Instruct,Base,2.0,200,0.677734375,0.175,0.995,0.98,0.975,0.346015625,0.6795673076923077,
character_substitution,Qwen2.5-14B-Instruct,Base,4.0,200,0.39421875,0.035,0.995,0.96,0.955,0.189765625,0.3997873036649215,
character_substitution,Qwen2.5-14B-Instruct,Base,8.0,200,0.292578125,0.005,0.89,0.92,0.83,0.127421875,0.3097703313253012,
character_substitution,Qwen2.5-14B-Instruct,Cellular_Bounds,2.0,760,0.8550986842105263,0.45921052631578946,0.9973684210526316,0.9289473684210526,0.9263157894736842,0.6045641447368421,0.8611505681818182,
character_substitution,Qwen2.5-14B-Instruct,Cellular_Bounds,4.0,760,0.5956825657894737,0.21973684210526315,0.9973684210526316,0.9184210526315789,0.9157894736842105,0.3438527960526316,0.5937051005747126,
character_substitution,Qwen2.5-14B-Instruct,Cellular_Bounds,8.0,760,0.4641241776315789,0.13289473684210526,0.9460526315789474,0.8881578947368421,0.8381578947368421,0.23961759868421054,0.4750294348508634,
character_substitution,Qwen2.5-32B-Instruct,Base,2.0,200,0.701953125,0.205,0.995,0.925,0.925,0.40359375,0.7125,
character_substitution,Qwen2.5-32B-Instruct,Base,4.0,200,0.4040625,0.045,1.0,0.88,0.88,0.19578125,0.42338423295454547,
character_substitution,Qwen2.5-32B-Instruct,Base,8.0,200,0.339140625,0.035,0.89,0.795,0.715,0.167265625,0.3924825174825175,
character_substitution,Qwen2.5-32B-Instruct,Cellular_Bounds,2.0,760,0.8811266447368421,0.5144736842105263,0.9907894736842106,0.9421052631578948,0.9328947368421052,0.6386513157894737,0.8861071932299013,
character_substitution,Qwen2.5-32B-Instruct,Cellular_Bounds,4.0,760,0.6849506578947369,0.25526315789473686,0.9986842105263158,0.9223684210526316,0.9210526315789473,0.3838815789473684,0.68546875,
character_substitution,Qwen2.5-32B-Instruct,Cellular_Bounds,8.0,760,0.5748149671052631,0.22631578947368422,0.925,0.9184210526315789,0.85,0.3258018092105263,0.5913312693498453,
character_substitution,Qwen2.5-7B-Instruct,Base,2.0,200,0.631875,0.14,0.995,0.89,0.885,0.314140625,0.6506002824858758,
character_substitution,Qwen2.5-7B-Instruct,Base,4.0,200,0.386640625,0.04,0.93,0.925,0.865,0.15953125,0.4093208092485549,
character_substitution,Qwen2.5-7B-Instruct,Base,8.0,200,0.242265625,0.005,0.805,0.91,0.73,0.103671875,0.265089897260274,
character_substitution,Qwen2.5-7B-Instruct,Cellular_Bounds,2.0,760,0.7605263157894737,0.33289473684210524,0.9960526315789474,0.7592105263157894,0.7578947368421053,0.49952713815789473,0.8224012586805556,
character_substitution,Qwen2.5-7B-Instruct,Cellular_Bounds,4.0,760,0.474609375,0.13026315789473683,0.9223684210526316,0.8052631578947368,0.7513157894736842,0.21872944078947368,0.5285683012259195,
character_substitution,Qwen2.5-7B-Instruct,Cellular_Bounds,8.0,760,0.37228618421052634,0.1,0.7052631578947368,0.8552631578947368,0.6171052631578947,0.18480674342105263,0.46005463752665243,
composed_permutation,Mistral-7B-Instruct-v0.3,Base,4.0,200,0.247578125,0.0,1.0,0.975,0.975,0.023671875,0.24783653846153847,
composed_permutation,Mistral-7B-Instruct-v0.3,Base,8.0,200,0.11125,0.0,0.99,0.845,0.835,0.017265625,0.11461452095808383,
composed_permutation,Mistral-7B-Instruct-v0.3,Cellular_Bounds,4.0,760,0.19664884868421054,0.0013157894736842105,0.9815789473684211,0.8802631578947369,0.8631578947368421,0.026151315789473686,0.19969512195121952,
composed_permutation,Mistral-7B-Instruct-v0.3,Cellular_Bounds,8.0,760,0.0966077302631579,0.0,0.9828947368421053,0.8486842105263158,0.8342105263157895,0.009806743421052632,0.09690457413249211,
composed_permutation,Phi-3.5-mini-instruct,Base,4.0,100,0.195,0.0,1.0,0.68,0.68,0.0128125,0.20519301470588236,
composed_permutation,Phi-3.5-mini-instruct,Base,8.0,100,0.1090625,0.0,1.0,0.55,0.55,0.01109375,0.11136363636363636,
composed_permutation,Phi-3.5-mini-instruct,Cellular_Bounds,4.0,380,0.2009046052631579,0.0,1.0,0.7552631578947369,0.7552631578947369,0.02528782894736842,0.20513937282229966,
composed_permutation,Phi-3.5-mini-instruct,Cellular_Bounds,8.0,380,0.11340460526315789,0.0,1.0,0.6131578947368421,0.6131578947368421,0.009169407894736843,0.1203057939914163,
composed_permutation,Qwen2.5-14B-Instruct,Base,4.0,200,0.268828125,0.0,0.995,0.99,0.985,0.042578125,0.2707011421319797,
composed_permutation,Qwen2.5-14B-Instruct,Base,8.0,200,0.1175,0.0,1.0,0.95,0.95,0.00765625,0.11850328947368421,
composed_permutation,Qwen2.5-14B-Instruct,Cellular_Bounds,4.0,760,0.24958881578947367,0.003947368421052632,1.0,0.9236842105263158,0.9236842105263158,0.027898848684210525,0.2532496438746439,
composed_permutation,Qwen2.5-14B-Instruct,Cellular_Bounds,8.0,760,0.11422697368421053,0.0013157894736842105,0.9973684210526316,0.8947368421052632,0.8921052631578947,0.009107730263157894,0.11810932890855458,
composed_permutation,Qwen2.5-32B-Instruct,Base,4.0,200,0.219765625,0.0,1.0,0.95,0.95,0.01421875,0.21957236842105263,
composed_permutation,Qwen2.5-32B-Instruct,Base,8.0,200,0.103671875,0.0,1.0,0.77,0.77,0.003984375,0.10399756493506493,
composed_permutation,Qwen2.5-32B-Instruct,Cellular_Bounds,4.0,760,0.1751233552631579,0.0,1.0,0.9210526315789473,0.9210526315789473,0.009189967105263158,0.17600446428571428,
composed_permutation,Qwen2.5-32B-Instruct,Cellular_Bounds,8.0,760,0.08671875,0.0,1.0,0.9052631578947369,0.9052631578947369,0.0028166118421052632,0.08795875726744186,
composed_permutation,Qwen2.5-7B-Instruct,Base,4.0,200,0.205390625,0.0,0.965,0.935,0.9,0.00671875,0.21024305555555556,
composed_permutation,Qwen2.5-7B-Instruct,Base,8.0,200,0.10859375,0.0,0.955,0.865,0.84,0.0115625,0.11467633928571429,
composed_permutation,Qwen2.5-7B-Instruct,Cellular_Bounds,4.0,760,0.1455797697368421,0.0,0.8947368421052632,0.825,0.7447368421052631,0.002734375,0.14898962014134276,
composed_permutation,Qwen2.5-7B-Instruct,Cellular_Bounds,8.0,760,0.07177220394736843,0.0,0.7605263157894737,0.8210526315789474,0.6328947368421053,0.0014802631578947369,0.07052364864864864,
conditional_shift,Mistral-7B-Instruct-v0.3,Base,2.0,200,0.51625,0.085,0.64,0.99,0.635,0.2525,0.5900590551181102,
conditional_shift,Mistral-7B-Instruct-v0.3,Base,4.0,200,0.320703125,0.005,0.755,0.945,0.71,0.093125,0.339568661971831,
conditional_shift,Mistral-7B-Instruct-v0.3,Base,8.0,200,0.2440625,0.0,0.795,0.97,0.775,0.0884375,0.27358870967741933,
conditional_shift,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2.0,760,0.5533717105263158,0.10394736842105264,0.6828947368421052,0.8263157894736842,0.531578947368421,0.27596628289473685,0.6583384900990099,
conditional_shift,Mistral-7B-Instruct-v0.3,Cellular_Bounds,4.0,760,0.3471217105263158,0.025,0.7776315789473685,0.8736842105263158,0.6723684210526316,0.13026315789473683,0.3854574363992172,
conditional_shift,Mistral-7B-Instruct-v0.3,Cellular_Bounds,8.0,760,0.2807154605263158,0.02236842105263158,0.7263157894736842,0.8697368421052631,0.6171052631578947,0.10024671052631579,0.33142324093816633,
conditional_shift,Phi-3.5-mini-instruct,Base,2.0,100,0.51328125,0.09,0.76,0.63,0.45,0.2209375,0.6170138888888889,
conditional_shift,Phi-3.5-mini-instruct,Base,4.0,100,0.27734375,0.01,0.85,0.53,0.46,0.0978125,0.32540760869565216,
conditional_shift,Phi-3.5-mini-instruct,Base,8.0,100,0.25140625,0.0,0.83,0.49,0.41,0.100625,0.34946646341463417,
conditional_shift,Phi-3.5-mini-instruct,Cellular_Bounds,2.0,380,0.44391447368421055,0.06842105263157895,0.4473684210526316,0.6657894736842105,0.32105263157894737,0.16430921052631578,0.6376793032786885,
conditional_shift,Phi-3.5-mini-instruct,Cellular_Bounds,4.0,380,0.3477796052631579,0.013157894736842105,0.7394736842105263,0.6447368421052632,0.49736842105263157,0.08486842105263158,0.4166666666666667,
conditional_shift,Phi-3.5-mini-instruct,Cellular_Bounds,8.0,380,0.3057565789473684,0.015789473684210527,0.7631578947368421,0.6763157894736842,0.5184210526315789,0.07709703947368421,0.3800761421319797,
conditional_shift,Qwen2.5-14B-Instruct,Base,2.0,100,0.583125,0.12,0.99,0.94,0.93,0.25953125,0.5917338709677419,
conditional_shift,Qwen2.5-14B-Instruct,Base,4.0,100,0.38796875,0.04,0.44,0.87,0.4,0.14921875,0.4953125,
conditional_shift,Qwen2.5-14B-Instruct,Base,8.0,100,0.308125,0.01,0.61,0.93,0.57,0.098125,0.3514254385964912,
conditional_shift,Qwen2.5-14B-Instruct,Cellular_Bounds,2.0,380,0.6627467105263158,0.22105263157894736,0.9815789473684211,0.8605263157894737,0.8421052631578947,0.3324013157894737,0.67626953125,
conditional_shift,Qwen2.5-14B-Instruct,Cellular_Bounds,4.0,380,0.5271792763157894,0.14210526315789473,0.5894736842105263,0.8710526315789474,0.5184210526315789,0.22393092105263157,0.65625,
conditional_shift,Qwen2.5-14B-Instruct,Cellular_Bounds,8.0,380,0.4479029605263158,0.034210526315789476,0.5052631578947369,0.8605263157894737,0.45,0.10855263157894737,0.5412097953216374,
conditional_shift,Qwen2.5-32B-Instruct,Base,2.0,100,0.6575,0.12,1.0,0.98,0.98,0.30078125,0.6615114795918368,
conditional_shift,Qwen2.5-32B-Instruct,Base,4.0,100,0.46640625,0.09,0.89,0.94,0.83,0.22234375,0.4924698795180723,
conditional_shift,Qwen2.5-32B-Instruct,Base,8.0,100,0.41734375,0.03,0.86,0.89,0.78,0.1534375,0.4577323717948718,
conditional_shift,Qwen2.5-32B-Instruct,Cellular_Bounds,2.0,380,0.8216694078947369,0.43157894736842106,0.9921052631578947,0.8894736842105263,0.881578947368421,0.5552631578947368,0.8429570895522388,
conditional_shift,Qwen2.5-32B-Instruct,Cellular_Bounds,4.0,380,0.6303453947368421,0.13421052631578947,0.8157894736842105,0.9052631578947369,0.7289473684210527,0.25884046052631576,0.665331678700361,
conditional_shift,Qwen2.5-32B-Instruct,Cellular_Bounds,8.0,380,0.5596628289473684,0.034210526315789476,0.6947368421052632,0.9289473684210526,0.6447368421052632,0.1359375,0.6023596938775511,
conditional_shift,Qwen2.5-7B-Instruct,Base,2.0,200,0.561328125,0.135,0.725,0.875,0.63,0.259453125,0.6715029761904762,
conditional_shift,Qwen2.5-7B-Instruct,Base,4.0,200,0.294375,0.02,0.42,0.89,0.37,0.075234375,0.37901182432432434,
conditional_shift,Qwen2.5-7B-Instruct,Base,8.0,200,0.231171875,0.005,0.485,0.825,0.445,0.033671875,0.26896067415730335,
conditional_shift,Qwen2.5-7B-Instruct,Cellular_Bounds,2.0,760,0.5391036184210526,0.13026315789473683,0.49078947368421055,0.8276315789473684,0.42236842105263156,0.24015213815789474,0.7207457165109035,
conditional_shift,Qwen2.5-7B-Instruct,Cellular_Bounds,4.0,760,0.3130345394736842,0.017105263157894738,0.2381578947368421,0.8236842105263158,0.2026315789473684,0.05633223684210526,0.44561688311688313,
conditional_shift,Qwen2.5-7B-Instruct,Cellular_Bounds,8.0,760,0.29950657894736843,0.003947368421052632,0.26842105263157895,0.8342105263157895,0.24473684210526317,0.03612253289473684,0.3880208333333333,
cyclic_shift,Mistral-7B-Instruct-v0.3,Base,2.0,280,0.5134068080357143,0.075,0.8392857142857143,0.9607142857142857,0.8142857142857143,0.19935825892857142,0.5620031524122807,
cyclic_shift,Mistral-7B-Instruct-v0.3,Base,4.0,280,0.3041434151785714,0.017857142857142856,0.42142857142857143,0.9642857142857143,0.4107142857142857,0.10721261160714286,0.3875,
cyclic_shift,Mistral-7B-Instruct-v0.3,Base,8.0,280,0.25046037946428573,0.010714285714285714,0.475,0.9642857142857143,0.45357142857142857,0.10357142857142858,0.3542691929133858,
cyclic_shift,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2.0,1160,0.49495554956896554,0.075,0.8241379310344827,0.7913793103448276,0.6413793103448275,0.18276535560344828,0.5446436071908602,
cyclic_shift,Mistral-7B-Instruct-v0.3,Cellular_Bounds,4.0,1160,0.31108230064655173,0.04224137931034483,0.4077586206896552,0.8017241379310345,0.30258620689655175,0.10972184806034482,0.480869391025641,
cyclic_shift,Mistral-7B-Instruct-v0.3,Cellular_Bounds,8.0,1160,0.3100518588362069,0.052586206896551725,0.47672413793103446,0.8344827586206897,0.3646551724137931,0.1359240301724138,0.49252918144208035,
cyclic_shift,Phi-3.5-mini-instruct,Base,2.0,140,0.4270089285714286,0.07142857142857142,1.0,0.5857142857142857,0.5857142857142857,0.16107700892857144,0.4423113567073171,
cyclic_shift,Phi-3.5-mini-instruct,Base,4.0,140,0.2731305803571429,0.02857142857142857,0.75,0.5285714285714286,0.37857142857142856,0.10390625,0.36224941037735847,
cyclic_shift,Phi-3.5-mini-instruct,Base,8.0,140,0.24436383928571428,0.02857142857142857,0.6428571428571429,0.55,0.39285714285714285,0.11071428571428571,0.33615056818181815,
cyclic_shift,Phi-3.5-mini-instruct,Cellular_Bounds,2.0,580,0.44349407327586204,0.0603448275862069,0.9586206896551724,0.6068965517241379,0.5758620689655173,0.15833108836206897,0.48566148952095806,
cyclic_shift,Phi-3.5-mini-instruct,Cellular_Bounds,4.0,580,0.2892780172413793,0.022413793103448276,0.45,0.6448275862068965,0.29482758620689653,0.08292699353448275,0.412874634502924,
cyclic_shift,Phi-3.5-mini-instruct,Cellular_Bounds,8.0,580,0.30766433189655173,0.03793103448275862,0.33275862068965517,0.6172413793103448,0.22758620689655173,0.11476966594827587,0.500295928030303,
cyclic_shift,Qwen2.5-14B-Instruct,Base,2.0,140,0.6221540178571429,0.12142857142857143,0.9857142857142858,0.9285714285714286,0.9142857142857143,0.269140625,0.63726806640625,
cyclic_shift,Qwen2.5-14B-Instruct,Base,4.0,140,0.36342075892857145,0.02857142857142857,0.09285714285714286,0.9928571428571429,0.09285714285714286,0.14185267857142858,0.7355769230769231,
cyclic_shift,Qwen2.5-14B-Instruct,Base,8.0,140,0.4085658482142857,0.06428571428571428,0.22142857142857142,1.0,0.22142857142857142,0.16498325892857144,0.7169858870967742,
cyclic_shift,Qwen2.5-14B-Instruct,Cellular_Bounds,2.0,580,0.7525188577586207,0.3120689655172414,0.9913793103448276,0.753448275862069,0.75,0.4461004849137931,0.801427801724138,
cyclic_shift,Qwen2.5-14B-Instruct,Cellular_Bounds,4.0,580,0.5552262931034483,0.15172413793103448,0.296551724137931,0.7948275862068965,0.19482758620689655,0.24929283405172414,0.9435840707964602,
cyclic_shift,Qwen2.5-14B-Instruct,Cellular_Bounds,8.0,580,0.573504849137931,0.12758620689655173,0.19137931034482758,0.7931034482758621,0.1793103448275862,0.2263806573275862,0.9251802884615384,
cyclic_shift,Qwen2.5-32B-Instruct,Base,2.0,140,0.6536551339285714,0.17142857142857143,1.0,0.9642857142857143,0.9642857142857143,0.30825892857142856,0.661255787037037,
cyclic_shift,Qwen2.5-32B-Instruct,Base,4.0,140,0.43761160714285713,0.07142857142857142,0.30714285714285716,0.95,0.30714285714285716,0.2011997767857143,0.7231104651162791,
cyclic_shift,Qwen2.5-32B-Instruct,Base,8.0,140,0.4419642857142857,0.08571428571428572,0.3,0.8428571428571429,0.29285714285714287,0.20599888392857144,0.7271341463414634,
cyclic_shift,Qwen2.5-32B-Instruct,Cellular_Bounds,2.0,580,0.7924164870689655,0.3396551724137931,0.9844827586206897,0.8241379310344827,0.8103448275862069,0.4896417025862069,0.834092420212766,
cyclic_shift,Qwen2.5-32B-Instruct,Cellular_Bounds,4.0,580,0.6448814655172413,0.21551724137931033,0.38448275862068965,0.7862068965517242,0.2775862068965517,0.3293238146551724,0.938858695652174,
cyclic_shift,Qwen2.5-32B-Instruct,Cellular_Bounds,8.0,580,0.6671605603448276,0.20689655172413793,0.33620689655172414,0.8068965517241379,0.24482758620689654,0.32502693965517243,0.9639084507042254,
cyclic_shift,Qwen2.5-7B-Instruct,Base,2.0,140,0.6018415178571429,0.10714285714285714,0.9857142857142858,0.8642857142857143,0.85,0.23275669642857144,0.6266084558823529,
cyclic_shift,Qwen2.5-7B-Instruct,Base,4.0,140,0.33203125,0.02857142857142857,0.12857142857142856,0.9642857142857143,0.12857142857142856,0.12734375,0.6625434027777778,
cyclic_shift,Qwen2.5-7B-Instruct,Base,8.0,140,0.3337332589285714,0.02857142857142857,0.2642857142857143,0.9,0.24285714285714285,0.12904575892857142,0.5685891544117647,
cyclic_shift,Qwen2.5-7B-Instruct,Cellular_Bounds,2.0,580,0.6556505926724138,0.23448275862068965,0.9879310344827587,0.5758620689655173,0.5689655172413793,0.35534078663793106,0.7927675189393939,
cyclic_shift,Qwen2.5-7B-Instruct,Cellular_Bounds,4.0,580,0.4282799030172414,0.10172413793103448,0.21379310344827587,0.6793103448275862,0.15689655172413794,0.19201912715517241,0.8756868131868132,
cyclic_shift,Qwen2.5-7B-Instruct,Cellular_Bounds,8.0,580,0.4771417025862069,0.11551724137931034,0.25,0.6982758620689655,0.20344827586206896,0.21067483836206896,0.8485169491525424,
random_permutation,Mistral-7B-Instruct-v0.3,Base,2.0,200,0.586796875,0.1,1.0,0.96,0.96,0.252890625,0.5909830729166666,
random_permutation,Mistral-7B-Instruct-v0.3,Base,4.0,200,0.330078125,0.015,1.0,0.935,0.935,0.113359375,0.33480949197860965,
random_permutation,Mistral-7B-Instruct-v0.3,Base,8.0,200,0.220859375,0.005,0.985,0.83,0.815,0.084765625,0.23773006134969324,
random_permutation,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2.0,760,0.6357113486842105,0.1868421052631579,0.9815789473684211,0.8671052631578947,0.85,0.3381578947368421,0.6568304953560371,
random_permutation,Mistral-7B-Instruct-v0.3,Cellular_Bounds,4.0,760,0.33102384868421053,0.02236842105263158,0.9842105263157894,0.8263157894736842,0.8105263157894737,0.10805921052631579,0.34681919642857145,
random_permutation,Mistral-7B-Instruct-v0.3,Cellular_Bounds,8.0,760,0.30267269736842106,0.02368421052631579,0.968421052631579,0.8065789473684211,0.7855263157894737,0.1052014802631579,0.32959066164154105,
random_permutation,Phi-3.5-mini-instruct,Base,2.0,100,0.5359375,0.13,1.0,0.74,0.74,0.24234375,0.5629222972972973,
random_permutation,Phi-3.5-mini-instruct,Base,4.0,100,0.28609375,0.01,1.0,0.57,0.57,0.0740625,0.33031798245614036,
random_permutation,Phi-3.5-mini-instruct,Base,8.0,100,0.15109375,0.0,1.0,0.47,0.47,0.02625,0.19448138297872342,
random_permutation,Phi-3.5-mini-instruct,Cellular_Bounds,2.0,380,0.4821546052631579,0.11578947368421053,1.0,0.7763157894736842,0.7763157894736842,0.22051809210526316,0.5045021186440678,
random_permutation,Phi-3.5-mini-instruct,Cellular_Bounds,4.0,380,0.30078125,0.02894736842105263,1.0,0.7684210526315789,0.7684210526315789,0.07401315789473684,0.3233090753424658,
random_permutation,Phi-3.5-mini-instruct,Cellular_Bounds,8.0,380,0.24235197368421052,0.018421052631578946,1.0,0.6552631578947369,0.6552631578947369,0.04860197368421053,0.28871736947791166,
random_permutation,Qwen2.5-14B-Instruct,Base,2.0,200,0.677734375,0.175,0.995,0.98,0.975,0.346015625,0.6795673076923077,
random_permutation,Qwen2.5-14B-Instruct,Base,4.0,200,0.363359375,0.03,0.995,0.975,0.97,0.1403125,0.3637242268041237,
random_permutation,Qwen2.5-14B-Instruct,Base,8.0,200,0.29375,0.005,0.975,0.965,0.94,0.112265625,0.3001994680851064,
random_permutation,Qwen2.5-14B-Instruct,Cellular_Bounds,2.0,760,0.8550986842105263,0.45921052631578946,0.9973684210526316,0.9289473684210526,0.9263157894736842,0.6045641447368421,0.8611505681818182,
random_permutation,Qwen2.5-14B-Instruct,Cellular_Bounds,4.0,760,0.5445518092105263,0.1631578947368421,0.9973684210526316,0.9131578947368421,0.9105263157894737,0.26194490131578946,0.5428332731213873,
random_permutation,Qwen2.5-14B-Instruct,Cellular_Bounds,8.0,760,0.4733552631578947,0.10657894736842105,0.9802631578947368,0.8881578947368421,0.8710526315789474,0.18692434210526315,0.4835253021148036,
random_permutation,Qwen2.5-32B-Instruct,Base,2.0,200,0.701953125,0.205,0.995,0.925,0.925,0.40359375,0.7125,
random_permutation,Qwen2.5-32B-Instruct,Base,4.0,200,0.44140625,0.055,0.99,0.895,0.885,0.192734375,0.4608050847457627,
random_permutation,Qwen2.5-32B-Instruct,Base,8.0,200,0.353984375,0.025,1.0,0.755,0.755,0.15640625,0.3971440397350993,
random_permutation,Qwen2.5-32B-Instruct,Cellular_Bounds,2.0,760,0.8811266447368421,0.5144736842105263,0.9907894736842106,0.9421052631578948,0.9328947368421052,0.6386513157894737,0.8861071932299013,
random_permutation,Qwen2.5-32B-Instruct,Cellular_Bounds,4.0,760,0.68046875,0.2578947368421053,0.9960526315789474,0.9157894736842105,0.9118421052631579,0.3796875,0.6820436507936508,
random_permutation,Qwen2.5-32B-Instruct,Cellular_Bounds,8.0,760,0.598828125,0.20789473684210527,0.9960526315789474,0.9105263157894737,0.9092105263157895,0.3164884868421053,0.60232000723589,
random_permutation,Qwen2.5-7B-Instruct,Base,2.0,200,0.62125,0.125,0.995,0.87,0.865,0.29890625,0.6452312138728323,
random_permutation,Qwen2.5-7B-Instruct,Base,4.0,200,0.373046875,0.04,0.885,0.895,0.785,0.15453125,0.4117237261146497,
random_permutation,Qwen2.5-7B-Instruct,Base,8.0,200,0.25671875,0.005,0.81,0.875,0.7,0.105234375,0.2875,
random_permutation,Qwen2.5-7B-Instruct,Cellular_Bounds,2.0,760,0.7584087171052631,0.33289473684210524,0.9947368421052631,0.7526315789473684,0.75,0.4954769736842105,0.8214364035087719,
random_permutation,Qwen2.5-7B-Instruct,Cellular_Bounds,4.0,760,0.4388157894736842,0.08157894736842106,0.8302631578947368,0.7907894736842105,0.6789473684210526,0.17380756578947368,0.501968265503876,
random_permutation,Qwen2.5-7B-Instruct,Cellular_Bounds,8.0,760,0.3864514802631579,0.08157894736842106,0.6934210526315789,0.8486842105263158,0.5894736842105263,0.1756373355263158,0.47938755580357145,
running_max,Mistral-7B-Instruct-v0.3,Base,2.0,200,0.63015625,0.15,0.995,0.615,0.61,0.35171875,0.6817366803278688,
running_max,Mistral-7B-Instruct-v0.3,Base,4.0,200,0.601484375,0.07,1.0,0.535,0.535,0.1728125,0.6629672897196262,
running_max,Mistral-7B-Instruct-v0.3,Base,8.0,200,0.482109375,0.035,1.0,0.685,0.685,0.0809375,0.4489051094890511,
running_max,Mistral-7B-Instruct-v0.3,Cellular_Bounds,2.0,760,0.5136101973684211,0.04473684210526316,0.9960526315789474,0.40657894736842104,0.4039473684210526,0.19354440789473684,0.6334995928338762,
running_max,Mistral-7B-Instruct-v0.3,Cellular_Bounds,4.0,760,0.4908511513157895,0.038157894736842106,1.0,0.39210526315789473,0.39210526315789473,0.11480263157894736,0.6356963087248322,
running_max,Mistral-7B-Instruct-v0.3,Cellular_Bounds,8.0,760,0.39847861842105264,0.021052631578947368,0.9947368421052631,0.4473684210526316,0.4421052631578947,0.07627467105263158,0.5046502976190477,
running_max,Qwen2.5-14B-Instruct,Base,2.0,200,0.979140625,0.895,1.0,0.995,0.995,0.95203125,0.9818624371859297,
running_max,Qwen2.5-14B-Instruct,Base,4.0,200,0.93671875,0.795,1.0,0.99,0.99,0.859453125,0.9400252525252525,
running_max,Qwen2.5-14B-Instruct,Base,8.0,200,0.84984375,0.515,1.0,0.995,0.995,0.6196875,0.851287688442211,
running_max,Qwen2.5-14B-Instruct,Cellular_Bounds,2.0,760,0.9139597039473685,0.5565789473684211,1.0,0.9,0.9,0.723889802631579,0.9243877923976608,
running_max,Qwen2.5-14B-Instruct,Cellular_Bounds,4.0,760,0.8521998355263158,0.425,1.0,0.8644736842105263,0.8644736842105263,0.5994860197368421,0.8676750380517504,
running_max,Qwen2.5-14B-Instruct,Cellular_Bounds,8.0,760,0.7801603618421052,0.33289473684210524,0.9986842105263158,0.8394736842105263,0.8381578947368421,0.508203125,0.8145849686028257,
running_max,Qwen2.5-32B-Instruct,Base,2.0,200,0.9975,0.98,1.0,0.985,0.985,0.9975,0.998730964467005,
running_max,Qwen2.5-32B-Instruct,Base,4.0,200,0.98953125,0.91,1.0,0.985,0.985,0.937265625,0.9912753807106599,
running_max,Qwen2.5-32B-Instruct,Base,8.0,200,0.950078125,0.705,1.0,0.97,0.97,0.79640625,0.9572326030927835,
running_max,Qwen2.5-32B-Instruct,Cellular_Bounds,2.0,760,0.9428659539473684,0.65,1.0,0.9447368421052632,0.9447368421052632,0.7873149671052632,0.9448337395543176,
running_max,Qwen2.5-32B-Instruct,Cellular_Bounds,4.0,760,0.8946340460526315,0.5407894736842105,1.0,0.9342105263157895,0.9342105263157895,0.7000205592105263,0.8950924295774648,
running_max,Qwen2.5-32B-Instruct,Cellular_Bounds,8.0,760,0.8637746710526316,0.4789473684210526,1.0,0.9342105263157895,0.9342105263157895,0.6497121710526316,0.8639524647887324,
running_max,Qwen2.5-7B-Instruct,Base,2.0,200,0.69859375,0.27,0.315,0.9,0.28,0.3253125,0.9866071428571429,
running_max,Qwen2.5-7B-Instruct,Base,4.0,200,0.783515625,0.275,0.795,0.805,0.62,0.432109375,0.873991935483871,
running_max,Qwen2.5-7B-Instruct,Base,8.0,200,0.609375,0.12,1.0,0.895,0.895,0.1653125,0.6363477653631285,
running_max,Qwen2.5-7B-Instruct,Cellular_Bounds,2.0,760,0.6951685855263158,0.3,0.38026315789473686,0.8302631578947368,0.3118421052631579,0.36893503289473684,0.9977584388185654,
running_max,Qwen2.5-7B-Instruct,Cellular_Bounds,4.0,760,0.6944284539473684,0.15,0.7789473684210526,0.7263157894736842,0.5657894736842105,0.28032483552631576,0.8071220930232558,
running_max,Qwen2.5-7B-Instruct,Cellular_Bounds,8.0,760,0.5842927631578947,0.07105263157894737,0.9526315789473684,0.7210526315789474,0.6947368421052632,0.18727384868421051,0.6498283617424242,
running_sum_local,Mistral-7B-Instruct-v0.3,Base,,200,0.502109375,0.015,1.0,0.785,0.785,0.1515625,0.5234872611464968,2.0
running_sum_local,Mistral-7B-Instruct-v0.3,Base,,200,0.270390625,0.0,0.565,0.78,0.415,0.10984375,0.3326430722891566,4.0
running_sum_local,Mistral-7B-Instruct-v0.3,Base,,200,0.1909375,0.0,0.695,0.795,0.515,0.10609375,0.23240291262135923,8.0
running_sum_local,Mistral-7B-Instruct-v0.3,Cellular_Bounds,,760,0.47633634868421054,0.003947368421052632,0.8947368421052632,0.6171052631578947,0.5592105263157895,0.15370065789473683,0.5491911764705882,2.0
running_sum_local,Mistral-7B-Instruct-v0.3,Cellular_Bounds,,760,0.24457236842105262,0.0013157894736842105,0.26052631578947366,0.7039473684210527,0.18421052631578946,0.09039884868421053,0.33203125,4.0
running_sum_local,Mistral-7B-Instruct-v0.3,Cellular_Bounds,,760,0.17384868421052632,0.0,0.42894736842105263,0.7460526315789474,0.3,0.08137335526315789,0.21957236842105263,8.0
running_sum_local,Phi-3.5-mini-instruct,Base,,100,0.57703125,0.06,0.99,0.94,0.93,0.24171875,0.5801411290322581,2.0
running_sum_local,Phi-3.5-mini-instruct,Base,,100,0.3253125,0.0,0.84,0.86,0.7,0.13921875,0.34866071428571427,4.0
running_sum_local,Phi-3.5-mini-instruct,Base,,100,0.239375,0.0,0.88,0.93,0.82,0.129375,0.24485518292682926,8.0
running_sum_local,Phi-3.5-mini-instruct,Cellular_Bounds,,380,0.5660361842105263,0.04473684210526316,0.9394736842105263,0.781578947368421,0.7421052631578947,0.19251644736842105,0.6100398936170213,2.0
running_sum_local,Phi-3.5-mini-instruct,Cellular_Bounds,,380,0.3040707236842105,0.005263157894736842,0.3078947368421053,0.6684210526315789,0.23421052631578948,0.12002467105263158,0.3967696629213483,4.0
running_sum_local,Phi-3.5-mini-instruct,Cellular_Bounds,,380,0.23211348684210525,0.013157894736842105,0.3684210526315789,0.6947368421052632,0.3105263157894737,0.1046875,0.2686705508474576,8.0
running_sum_local,Qwen2.5-14B-Instruct,Base,,200,0.7475,0.215,1.0,0.855,0.855,0.47796875,0.7768640350877193,2.0
running_sum_local,Qwen2.5-14B-Instruct,Base,,200,0.5134375,0.085,0.825,0.765,0.65,0.304375,0.5865384615384616,4.0
running_sum_local,Qwen2.5-14B-Instruct,Base,,200,0.38203125,0.04,0.58,0.72,0.49,0.241875,0.509406887755102,8.0
running_sum_local,Qwen2.5-14B-Instruct,Cellular_Bounds,,760,0.7080797697368421,0.07631578947368421,1.0,0.8947368421052632,0.8947368421052632,0.30803865131578945,0.7241038602941177,2.0
running_sum_local,Qwen2.5-14B-Instruct,Cellular_Bounds,,760,0.526110197368421,0.05131578947368421,0.6802631578947368,0.868421052631579,0.6118421052631579,0.22376644736842105,0.5924059139784946,4.0
running_sum_local,Qwen2.5-14B-Instruct,Cellular_Bounds,,760,0.4118215460526316,0.034210526315789476,0.44605263157894737,0.8447368421052631,0.4144736842105263,0.1901110197368421,0.5431051587301587,8.0
running_sum_local,Qwen2.5-32B-Instruct,Base,,200,0.76796875,0.225,1.0,0.945,0.945,0.44765625,0.7788525132275133,2.0
running_sum_local,Qwen2.5-32B-Instruct,Base,,200,0.546484375,0.08,0.885,0.885,0.795,0.3165625,0.5835298742138365,4.0
running_sum_local,Qwen2.5-32B-Instruct,Base,,200,0.45609375,0.045,0.605,0.87,0.555,0.247734375,0.5677083333333334,8.0
running_sum_local,Qwen2.5-32B-Instruct,Cellular_Bounds,,760,0.77734375,0.08947368421052632,1.0,0.9460526315789474,0.9460526315789474,0.3298314144736842,0.7830102573018081,2.0
running_sum_local,Qwen2.5-32B-Instruct,Cellular_Bounds,,760,0.5734786184210526,0.039473684210526314,0.7,0.9473684210526315,0.6881578947368421,0.23787006578947367,0.6240738527724665,4.0
running_sum_local,Qwen2.5-32B-Instruct,Cellular_Bounds,,760,0.45279605263157896,0.018421052631578946,0.3973684210526316,0.9447368421052632,0.39210526315789473,0.17407483552631578,0.5656984060402684,8.0
running_sum_local,Qwen2.5-7B-Instruct,Base,,200,0.4125,0.0,0.675,0.42,0.285,0.04453125,0.5512609649122807,2.0
running_sum_local,Qwen2.5-7B-Instruct,Base,,200,0.278515625,0.005,0.89,0.42,0.38,0.11046875,0.38363486842105265,4.0
running_sum_local,Qwen2.5-7B-Instruct,Base,,200,0.2365625,0.0,0.98,0.45,0.45,0.132578125,0.3345486111111111,8.0
running_sum_local,Qwen2.5-7B-Instruct,Cellular_Bounds,,760,0.3911595394736842,0.003947368421052632,0.6026315789473684,0.5434210526315789,0.3078947368421053,0.06527549342105263,0.4953258547008547,2.0
running_sum_local,Qwen2.5-7B-Instruct,Cellular_Bounds,,760,0.26116365131578945,0.002631578947368421,0.6263157894736842,0.5921052631578947,0.36973684210526314,0.0739514802631579,0.31817170818505336,4.0
running_sum_local,Qwen2.5-7B-Instruct,Cellular_Bounds,,760,0.1846217105263158,0.0013157894736842105,0.5381578947368421,0.6078947368421053,0.34210526315789475,0.07545230263157894,0.24326923076923077,8.0
```

---

## Appendix H: Gemini per-length x per-density breakdowns

# Gemini (gemini-flash-latest) per-length x per-density breakdown

Source: `results_clean/gemini_pilot/gemini-flash-latest_POOLED_3trials.csv`, n=3 trials/cell, k=3 excluded (standard convention).


## character_substitution

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 0.833 | 0.083 | 0.667 | 0.167 | 9 |
| 4 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 4 | Cellular_Bounds | 2 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 8 | Base | 0 | 0.875 | 0.083 | 0.778 | 0.147 | 9 |
| 8 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 8 | Cellular_Bounds | 2 | 0.944 | 0.030 | 0.667 | 0.167 | 9 |
| 8 | Cellular_Bounds | 4 | 0.917 | 0.055 | 0.667 | 0.167 | 9 |
| 16 | Base | 0 | 0.729 | 0.065 | 0.111 | 0.111 | 9 |
| 16 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 2 | 0.993 | 0.007 | 0.889 | 0.111 | 9 |
| 16 | Cellular_Bounds | 4 | 0.840 | 0.066 | 0.444 | 0.176 | 9 |
| 16 | Cellular_Bounds | 8 | 0.688 | 0.067 | 0.000 | 0.000 | 9 |
| 32 | Base | 0 | 0.517 | 0.028 | 0.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 2 | 0.972 | 0.014 | 0.556 | 0.176 | 9 |
| 32 | Cellular_Bounds | 4 | 0.882 | 0.050 | 0.333 | 0.167 | 9 |
| 32 | Cellular_Bounds | 8 | 0.736 | 0.056 | 0.111 | 0.111 | 9 |
| 32 | Cellular_Bounds | 16 | 0.708 | 0.040 | 0.000 | 0.000 | 9 |
| 64 | Base | 0 | 0.406 | 0.043 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 2 | 0.969 | 0.018 | 0.667 | 0.167 | 9 |
| 64 | Cellular_Bounds | 4 | 0.889 | 0.033 | 0.111 | 0.111 | 9 |
| 64 | Cellular_Bounds | 8 | 0.726 | 0.047 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 16 | 0.608 | 0.047 | 0.000 | 0.000 | 9 |

## composed_permutation

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 0.167 | 0.083 | 0.000 | 0.000 | 6 |
| 4 | Cellular_Bounds | 1 | 0.500 | 0.158 | 0.167 | 0.167 | 6 |
| 4 | Cellular_Bounds | 2 | 0.125 | 0.085 | 0.000 | 0.000 | 6 |
| 8 | Base | 0 | 0.125 | 0.046 | 0.000 | 0.000 | 6 |
| 8 | Cellular_Bounds | 1 | 0.521 | 0.146 | 0.167 | 0.167 | 6 |
| 8 | Cellular_Bounds | 2 | 0.062 | 0.043 | 0.000 | 0.000 | 6 |
| 8 | Cellular_Bounds | 4 | 0.208 | 0.070 | 0.000 | 0.000 | 6 |
| 16 | Base | 0 | 0.052 | 0.052 | 0.000 | 0.000 | 6 |
| 16 | Cellular_Bounds | 1 | 0.146 | 0.146 | 0.000 | 0.000 | 6 |
| 16 | Cellular_Bounds | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 16 | Cellular_Bounds | 4 | 0.062 | 0.043 | 0.000 | 0.000 | 6 |
| 16 | Cellular_Bounds | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 32 | Base | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 32 | Cellular_Bounds | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 32 | Cellular_Bounds | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 32 | Cellular_Bounds | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 32 | Cellular_Bounds | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 32 | Cellular_Bounds | 16 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 64 | Base | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 64 | Cellular_Bounds | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 64 | Cellular_Bounds | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 64 | Cellular_Bounds | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 64 | Cellular_Bounds | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| 64 | Cellular_Bounds | 16 | 0.000 | 0.000 | 0.000 | 0.000 | 6 |

## conditional_shift

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 0.917 | 0.059 | 0.778 | 0.147 | 9 |
| 4 | Cellular_Bounds | 1 | 0.972 | 0.028 | 0.889 | 0.111 | 9 |
| 4 | Cellular_Bounds | 2 | 0.944 | 0.056 | 0.889 | 0.111 | 9 |
| 8 | Base | 0 | 0.806 | 0.066 | 0.333 | 0.167 | 9 |
| 8 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 8 | Cellular_Bounds | 2 | 0.917 | 0.029 | 0.444 | 0.176 | 9 |
| 8 | Cellular_Bounds | 4 | 0.819 | 0.073 | 0.444 | 0.176 | 9 |
| 16 | Base | 0 | 0.625 | 0.091 | 0.111 | 0.111 | 9 |
| 16 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 2 | 0.868 | 0.110 | 0.778 | 0.147 | 9 |
| 16 | Cellular_Bounds | 4 | 0.903 | 0.018 | 0.111 | 0.111 | 9 |
| 16 | Cellular_Bounds | 8 | 0.569 | 0.100 | 0.000 | 0.000 | 9 |
| 32 | Base | 0 | 0.240 | 0.120 | 0.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 1 | 0.556 | 0.176 | 0.556 | 0.176 | 9 |
| 32 | Cellular_Bounds | 2 | 0.552 | 0.175 | 0.444 | 0.176 | 9 |
| 32 | Cellular_Bounds | 4 | 0.479 | 0.166 | 0.333 | 0.167 | 9 |
| 32 | Cellular_Bounds | 8 | 0.497 | 0.135 | 0.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 16 | 0.323 | 0.131 | 0.000 | 0.000 | 9 |
| 64 | Base | 0 | 0.132 | 0.087 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 1 | 0.667 | 0.167 | 0.667 | 0.167 | 9 |
| 64 | Cellular_Bounds | 2 | 0.111 | 0.111 | 0.111 | 0.111 | 9 |
| 64 | Cellular_Bounds | 4 | 0.215 | 0.142 | 0.111 | 0.111 | 9 |
| 64 | Cellular_Bounds | 8 | 0.325 | 0.144 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 16 | 0.215 | 0.111 | 0.000 | 0.000 | 9 |

## copy_previous

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 4 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 4 | Cellular_Bounds | 2 | 0.917 | 0.083 | 0.667 | 0.333 | 3 |
| 8 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 8 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 8 | Cellular_Bounds | 2 | 0.917 | 0.042 | 0.333 | 0.333 | 3 |
| 8 | Cellular_Bounds | 4 | 0.958 | 0.042 | 0.667 | 0.333 | 3 |
| 16 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 16 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 16 | Cellular_Bounds | 2 | 0.896 | 0.021 | 0.000 | 0.000 | 3 |
| 16 | Cellular_Bounds | 4 | 0.896 | 0.055 | 0.333 | 0.333 | 3 |
| 16 | Cellular_Bounds | 8 | 0.958 | 0.021 | 0.333 | 0.333 | 3 |
| 32 | Base | 0 | 0.938 | 0.062 | 0.667 | 0.333 | 3 |
| 32 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 32 | Cellular_Bounds | 2 | 0.802 | 0.063 | 0.000 | 0.000 | 3 |
| 32 | Cellular_Bounds | 4 | 0.875 | 0.065 | 0.333 | 0.333 | 3 |
| 32 | Cellular_Bounds | 8 | 0.948 | 0.028 | 0.333 | 0.333 | 3 |
| 32 | Cellular_Bounds | 16 | 0.979 | 0.010 | 0.333 | 0.333 | 3 |
| 64 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 2 | 0.604 | 0.038 | 0.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 4 | 0.875 | 0.016 | 0.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 8 | 0.938 | 0.009 | 0.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 16 | 0.990 | 0.005 | 0.333 | 0.333 | 3 |

## cyclic_shift

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 4 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 4 | Cellular_Bounds | 2 | 0.972 | 0.028 | 0.889 | 0.111 | 9 |
| 8 | Base | 0 | 0.944 | 0.030 | 0.667 | 0.167 | 9 |
| 8 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 8 | Cellular_Bounds | 2 | 0.972 | 0.018 | 0.778 | 0.147 | 9 |
| 8 | Cellular_Bounds | 4 | 0.931 | 0.030 | 0.556 | 0.176 | 9 |
| 16 | Base | 0 | 0.924 | 0.027 | 0.444 | 0.176 | 9 |
| 16 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 2 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 4 | 0.986 | 0.014 | 0.889 | 0.111 | 9 |
| 16 | Cellular_Bounds | 8 | 0.861 | 0.043 | 0.111 | 0.111 | 9 |
| 32 | Base | 0 | 0.590 | 0.022 | 0.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 2 | 0.986 | 0.011 | 0.778 | 0.147 | 9 |
| 32 | Cellular_Bounds | 4 | 0.941 | 0.015 | 0.222 | 0.147 | 9 |
| 32 | Cellular_Bounds | 8 | 0.872 | 0.048 | 0.222 | 0.147 | 9 |
| 32 | Cellular_Bounds | 16 | 0.719 | 0.059 | 0.000 | 0.000 | 9 |
| 64 | Base | 0 | 0.517 | 0.033 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 2 | 0.988 | 0.006 | 0.556 | 0.176 | 9 |
| 64 | Cellular_Bounds | 4 | 0.929 | 0.026 | 0.333 | 0.167 | 9 |
| 64 | Cellular_Bounds | 8 | 0.788 | 0.038 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 16 | 0.736 | 0.042 | 0.000 | 0.000 | 9 |
| 128 | Base | 0 | 0.363 | 0.032 | 0.000 | 0.000 | 9 |
| 128 | Cellular_Bounds | 1 | 0.766 | 0.000 | 0.000 | 0.000 | 9 |
| 128 | Cellular_Bounds | 2 | 0.881 | 0.088 | 0.333 | 0.167 | 9 |
| 128 | Cellular_Bounds | 4 | 0.862 | 0.057 | 0.222 | 0.147 | 9 |
| 128 | Cellular_Bounds | 8 | 0.709 | 0.065 | 0.000 | 0.000 | 9 |
| 128 | Cellular_Bounds | 16 | 0.602 | 0.082 | 0.000 | 0.000 | 9 |
| 256 | Base | 0 | 0.278 | 0.048 | 0.000 | 0.000 | 9 |
| 256 | Cellular_Bounds | 1 | 0.383 | 0.000 | 0.000 | 0.000 | 9 |
| 256 | Cellular_Bounds | 2 | 0.732 | 0.008 | 0.000 | 0.000 | 9 |
| 256 | Cellular_Bounds | 4 | 0.833 | 0.052 | 0.222 | 0.147 | 9 |
| 256 | Cellular_Bounds | 8 | 0.625 | 0.083 | 0.000 | 0.000 | 9 |
| 256 | Cellular_Bounds | 16 | 0.621 | 0.060 | 0.000 | 0.000 | 9 |

## random_permutation

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 0.778 | 0.114 | 0.667 | 0.167 | 9 |
| 4 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 4 | Cellular_Bounds | 2 | 0.944 | 0.037 | 0.778 | 0.147 | 9 |
| 8 | Base | 0 | 0.792 | 0.078 | 0.444 | 0.176 | 9 |
| 8 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 8 | Cellular_Bounds | 2 | 0.917 | 0.069 | 0.778 | 0.147 | 9 |
| 8 | Cellular_Bounds | 4 | 0.847 | 0.068 | 0.556 | 0.176 | 9 |
| 16 | Base | 0 | 0.556 | 0.099 | 0.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 2 | 0.993 | 0.007 | 0.889 | 0.111 | 9 |
| 16 | Cellular_Bounds | 4 | 0.868 | 0.050 | 0.444 | 0.176 | 9 |
| 16 | Cellular_Bounds | 8 | 0.646 | 0.078 | 0.000 | 0.000 | 9 |
| 32 | Base | 0 | 0.490 | 0.037 | 0.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 2 | 0.924 | 0.035 | 0.444 | 0.176 | 9 |
| 32 | Cellular_Bounds | 4 | 0.778 | 0.058 | 0.333 | 0.167 | 9 |
| 32 | Cellular_Bounds | 8 | 0.615 | 0.081 | 0.111 | 0.111 | 9 |
| 32 | Cellular_Bounds | 16 | 0.618 | 0.052 | 0.000 | 0.000 | 9 |
| 64 | Base | 0 | 0.413 | 0.030 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 2 | 0.967 | 0.015 | 0.556 | 0.176 | 9 |
| 64 | Cellular_Bounds | 4 | 0.762 | 0.075 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 8 | 0.613 | 0.066 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 16 | 0.576 | 0.040 | 0.000 | 0.000 | 9 |

## running_max

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 4 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 4 | Cellular_Bounds | 2 | 0.889 | 0.061 | 0.667 | 0.167 | 9 |
| 8 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 8 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 8 | Cellular_Bounds | 2 | 0.722 | 0.090 | 0.333 | 0.167 | 9 |
| 8 | Cellular_Bounds | 4 | 0.750 | 0.072 | 0.222 | 0.147 | 9 |
| 16 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 2 | 0.903 | 0.077 | 0.778 | 0.147 | 9 |
| 16 | Cellular_Bounds | 4 | 0.715 | 0.063 | 0.111 | 0.111 | 9 |
| 16 | Cellular_Bounds | 8 | 0.785 | 0.072 | 0.333 | 0.167 | 9 |
| 32 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 2 | 0.872 | 0.082 | 0.667 | 0.167 | 9 |
| 32 | Cellular_Bounds | 4 | 0.667 | 0.071 | 0.111 | 0.111 | 9 |
| 32 | Cellular_Bounds | 8 | 0.809 | 0.031 | 0.111 | 0.111 | 9 |
| 32 | Cellular_Bounds | 16 | 0.906 | 0.036 | 0.333 | 0.167 | 9 |
| 64 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 2 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 4 | 0.580 | 0.099 | 0.222 | 0.147 | 9 |
| 64 | Cellular_Bounds | 8 | 0.653 | 0.071 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 16 | 0.854 | 0.034 | 0.000 | 0.000 | 9 |

## running_sum_local

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 9 |
| 4 | Cellular_Bounds | 1 | 0.667 | 0.102 | 0.333 | 0.167 | 9 |
| 4 | Cellular_Bounds | 2 | 0.917 | 0.042 | 0.667 | 0.167 | 9 |
| 8 | Base | 0 | 0.986 | 0.014 | 0.889 | 0.111 | 9 |
| 8 | Cellular_Bounds | 1 | 0.708 | 0.137 | 0.556 | 0.176 | 9 |
| 8 | Cellular_Bounds | 2 | 0.722 | 0.040 | 0.000 | 0.000 | 9 |
| 8 | Cellular_Bounds | 4 | 0.889 | 0.014 | 0.111 | 0.111 | 9 |
| 16 | Base | 0 | 0.972 | 0.011 | 0.556 | 0.176 | 9 |
| 16 | Cellular_Bounds | 1 | 0.806 | 0.123 | 0.667 | 0.167 | 9 |
| 16 | Cellular_Bounds | 2 | 0.688 | 0.043 | 0.000 | 0.000 | 9 |
| 16 | Cellular_Bounds | 4 | 0.882 | 0.022 | 0.111 | 0.111 | 9 |
| 16 | Cellular_Bounds | 8 | 0.958 | 0.010 | 0.333 | 0.167 | 9 |
| 32 | Base | 0 | 0.941 | 0.023 | 0.444 | 0.176 | 9 |
| 32 | Cellular_Bounds | 1 | 0.528 | 0.147 | 0.333 | 0.167 | 9 |
| 32 | Cellular_Bounds | 2 | 0.656 | 0.039 | 0.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 4 | 0.830 | 0.020 | 0.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 8 | 0.924 | 0.017 | 0.000 | 0.000 | 9 |
| 32 | Cellular_Bounds | 16 | 0.948 | 0.016 | 0.222 | 0.147 | 9 |
| 64 | Base | 0 | 0.780 | 0.044 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 1 | 0.866 | 0.097 | 0.222 | 0.147 | 9 |
| 64 | Cellular_Bounds | 2 | 0.618 | 0.057 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 4 | 0.826 | 0.018 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 8 | 0.903 | 0.015 | 0.000 | 0.000 | 9 |
| 64 | Cellular_Bounds | 16 | 0.920 | 0.014 | 0.000 | 0.000 | 9 |

## running_xor

| Seq_Length | Method | k | Mean TAR | SEM | Mean SAR | SEM | n |
|---|---|---|---|---|---|---|---|
| 4 | Base | 0 | 0.917 | 0.083 | 0.667 | 0.333 | 3 |
| 4 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 4 | Cellular_Bounds | 2 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 8 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 8 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 8 | Cellular_Bounds | 2 | 0.792 | 0.083 | 0.000 | 0.000 | 3 |
| 8 | Cellular_Bounds | 4 | 0.917 | 0.042 | 0.333 | 0.333 | 3 |
| 16 | Base | 0 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 16 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 16 | Cellular_Bounds | 2 | 0.688 | 0.036 | 0.000 | 0.000 | 3 |
| 16 | Cellular_Bounds | 4 | 0.875 | 0.036 | 0.000 | 0.000 | 3 |
| 16 | Cellular_Bounds | 8 | 0.979 | 0.021 | 0.667 | 0.333 | 3 |
| 32 | Base | 0 | 0.958 | 0.028 | 0.333 | 0.333 | 3 |
| 32 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 32 | Cellular_Bounds | 2 | 0.802 | 0.055 | 0.000 | 0.000 | 3 |
| 32 | Cellular_Bounds | 4 | 0.844 | 0.000 | 0.000 | 0.000 | 3 |
| 32 | Cellular_Bounds | 8 | 0.948 | 0.028 | 0.333 | 0.333 | 3 |
| 32 | Cellular_Bounds | 16 | 0.990 | 0.010 | 0.667 | 0.333 | 3 |
| 64 | Base | 0 | 0.625 | 0.313 | 0.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 1 | 1.000 | 0.000 | 1.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 2 | 0.750 | 0.009 | 0.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 4 | 0.828 | 0.009 | 0.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 8 | 0.953 | 0.009 | 0.000 | 0.000 | 3 |
| 64 | Cellular_Bounds | 16 | 0.984 | 0.009 | 0.333 | 0.333 | 3 |
