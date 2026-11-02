/**
 * Raw ISA interpreter for 2..30-bit words, including page 2 and vector ops.
 * tests/test_js_interpreter.py compares it with strandedcpu/interpreter.py.
 */

// VRAND hash constants per N: [c[0..N-1], K, h1, h2] (strandedcpu.transformer.vrand_constants)
const VRAND_CONSTANTS = {"2":[[3,3],3,1,1],"3":[[7,5,7],7,1,1],"4":[[15,7,13,15],13,1,2],"5":[[29,13,25,29,15],27,1,2],"6":[[55,25,49,57,27,3],53,2,3],"7":[[109,51,99,115,55,7,35],105,2,3],"8":[[217,99,195,229,109,11,67,249],211,3,4],"9":[[433,199,389,457,217,21,133,495,263],423,3,4],"10":[[865,395,777,913,431,43,267,989,525,499],845,3,5],"11":[[1731,789,1553,1825,863,83,531,1979,1049,997,831],1689,4,5],"12":[[3459,1579,3105,3647,1723,167,1061,3955,2095,1991,1659,3763],3377,4,6],"13":[[6919,3157,6211,7293,3447,333,2123,7911,4189,3981,3319,7523,6421],6753,4,6],"14":[[13837,6313,12419,14587,6891,665,4243,15819,8377,7963,6635,15045,12843,13597],13505,5,7],"15":[[27671,12625,24837,29173,13783,1327,8485,31637,16755,15923,13269,30089,25685,27193,9939],27007,5,7],"16":[[55341,25249,49675,58345,27563,2655,16969,63273,33507,31847,26539,60179,51369,54387,19879,63427],54015,6,8],"17":[[110681,50495,99347,116687,55127,5307,33937,126547,67015,63693,53077,120355,102735,108771,39757,126853,62469],108031,6,8],"18":[[221361,100991,198695,233373,110251,10613,67875,253091,134029,127385,106153,240711,205469,217541,79513,253703,124939,93861],216059,6,9],"19":[[442721,201981,397387,466745,220501,21227,135747,506183,268057,254767,212303,481421,410937,435083,159025,507407,249875,187721,305861],432119,7,9],"20":[[885441,403959,794773,933489,441003,42451,271495,1012365,536111,509533,424605,962839,821873,870165,318047,1014813,499749,375443,611721,934975],864237,7,10],"21":[[1770881,807919,1589547,1866977,882003,84903,542989,2024727,1072221,1019065,849209,1925677,1643745,1740329,636093,2029625,999497,750885,1223441,1869949,1904451],1728473,7,10],"22":[[3541763,1615837,3179093,3733955,1764007,169805,1085977,4049455,2144443,2038129,1698417,3851355,3287491,3480657,1272187,4059249,1998993,1501769,2446883,3739897,3808903,916217],3456945,8,11],"23":[[7083525,3231673,6358183,7467909,3528011,339609,2171953,8098907,4288885,4076257,3396835,7702709,6574981,6961311,2544373,8118495,3997987,3003537,4893765,7479793,7617805,1832431,4233621],6913891,8,11],"24":[[14167049,6463345,12716365,14935815,7056021,679217,4343903,16197813,8577767,8152515,6793669,15405417,13149959,13922623,5088745,16236991,7995971,6007073,9787527,14959585,15235607,3664861,8467241,2336627],13827781,9,12],"25":[[28334097,12926689,25432731,29871631,14112041,1358431,8687805,32395627,17155533,16305029,13587335,30810833,26299917,27845243,10177487,32473981,15991941,12014145,19575053,29919167,30471213,7329723,16934481,4673251,9456909],27655563,9,12],"26":[[56668193,25853375,50865459,59743259,28224083,2716861,17375609,64791253,34311067,32610057,27174671,61621663,52599833,55690487,20354973,64947959,31983881,24028287,39150107,59838335,60942425,14659445,33868963,9346501,18913819,9378181],55311127,9,13],"27":[[113336383,51706751,101730919,119486517,56448163,5433723,34751219,129582505,68622133,65220113,54349341,123243327,105199667,111380973,40709945,129895917,63967761,48056575,78300213,119676669,121884849,29318887,67737925,18693001,37827637,18756363,101442325],110622251,10,13],"28":[[226672765,103413501,203461835,238973033,112896327,10867445,69502437,259165009,137244263,130440223,108698681,246486653,210399333,222761943,81419889,259791833,127935521,96113147,156600425,239353337,243769699,58637775,135475849,37386003,75655273,37512723,202884649,25455939],221244503,10,14],"29":[[453345531,206827001,406923671,477946065,225792651,21734887,139004873,518330017,274488525,260880445,217397361,492973303,420798665,445523885,162839779,519583665,255871041,192226295,313200849,478706673,487539395,117275549,270951695,74772005,151310543,75025447,405769297,50911877,331984329],442489007,10,14],"30":[[906691061,413654001,813847341,955892129,451585303,43469775,278009743,1036660031,548977049,521760891,434794719,985946605,841597327,891047769,325679555,1039167331,511742083,384452589,626401697,957413343,975078789,234551095,541903391,149544007,302621085,150050893,811538591,101823755,663968657,858351977],884978011,11,15]};

function logaddTable(fracBits) {
    // T[d] = round(2^F * log2(1 + 2^(-d / 2^F))), ending before the first 0
    const scale = 2 ** fracBits, table = [];
    for (;;) {
        const v = Math.floor(scale * Math.log2(1 + 2 ** (-table.length / scale)) + 0.5);
        if (v === 0) return table;
        table.push(v);
    }
}

const PAGE2 = new Set([64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 84, 85, 86, 87, 88]);


class ISAInterpreter {
    /**
     * @param {number} s - scratchpad columns (address offset)
     * @param {number} m - memory slots
     * @param {number} N - bits per value
     * @param {number[]} initialMemory - initial memory values (length m)
     * @param {number[][]} commands - array of [op, b, c] instruction triples
     * @param {number} logFracBits - LOGADD fraction bits (strandedcpu.transformer.Config.log_frac_bits)
     */
    constructor(s, m, N, initialMemory, commands, logFracBits = 2) {
        this.s = s;
        this.m = m;
        this.N = N;
        this.mod = 2 ** N;
        this.mask = this.mod - 1;
        this.half = 2 ** (N - 1);
        this.commands = commands;
        this.cmdStart = s + m;
        this.memory = new Array(m);
        for (let i = 0; i < m; i++) this.memory[i] = this.clamp(initialMemory[i] || 0);
        this.pc = this.cmdStart; // start at first instruction
        this.carry = 0;
        this.vlLog = 0;
        this.vcarry = new Array(m).fill(0);
        this.table = logaddTable(logFracBits);
    }

    clamp(val) {
        val = ((val % this.mod) + this.mod) % this.mod;
        return val >= this.half ? val - this.mod : val;
    }

    unsigned(val) {
        return ((val % this.mod) + this.mod) % this.mod;
    }

    getMem(addr) {
        const idx = addr - this.s;
        if (idx < 0 || idx >= this.m) return 0;
        return this.memory[idx];
    }

    setMem(addr, val) {
        const idx = addr - this.s;
        if (idx >= 0 && idx < this.m) this.memory[idx] = this.clamp(val);
    }

    /** LOAD/STORE address: pointers reach memory word p mod s. */
    pointer(p) {
        return this.s + this.unsigned(p) % this.s;
    }

    /** Set PC to the first instruction; preserve memory and flags. */
    resetPC() {
        this.pc = this.cmdStart;
    }

    logadd(b, c) {
        const d = Math.abs(b - c);
        return this.clamp(Math.max(b, c) + (d < this.table.length ? this.table[d] : 0));
    }

    shift(b, c) {
        if (c >= 0) return this.clamp(b * 2 ** Math.min(c, this.N + 1));
        return Math.floor(b / 2 ** Math.min(-c, this.N + 1));  // arithmetic >>
    }

    xlatAddress(v, cCol) {
        const k = Math.floor(Math.log2(this.s));
        const low = this.unsigned(v) % 2 ** Math.min(k, this.N);
        return cCol | low;
    }

    addCarry(b, c) { return Math.floor((this.unsigned(b) + this.unsigned(c)) / this.mod); }
    subCarry(b, c) { return this.unsigned(b) >= this.unsigned(c) ? 1 : 0; }

    /** Execute one instruction. Returns true if halted. */
    step() {
        if (this.pc === 0) return true;

        const instrIdx = this.pc - this.cmdStart;
        if (instrIdx < 0 || instrIdx >= this.commands.length) {
            this.pc = 0;
            return true;
        }

        const cmd = this.commands[instrIdx];
        const op = cmd[0], b = cmd[1], c = cmd[2];
        const N = this.N;
        this.pc++; // default: advance
        const vb = this.getMem(b), vc = this.getMem(c);

        if (op >= 32 && !PAGE2.has(op)) {          // SUBLEQ: mem[b] -= mem[a]; jump if <= 0
            const r = this.clamp(vb - this.getMem(op));
            this.setMem(b, r);
            if (r <= 0) this.pc = c;
            return false;
        }
        switch (op) {
            case 0: this.pc = 0; return true;                         // HALT
            case 1: this.setMem(b, vc); break;                        // MOV
            case 2: this.carry = this.addCarry(vb, vc); this.setMem(b, vb + vc); break;   // ADD
            case 3: this.pc = c; break;                               // JMP
            case 4: if (vb === 0) this.pc = c; break;                 // JZ
            case 5: if (vb !== 0) this.pc = c; break;                 // JNZ
            case 6: this.setMem(b, vb + 1); break;                    // INC
            case 7: this.setMem(b, vb - 1); break;                    // DEC
            case 8: this.setMem(b, vb * 2); break;                    // SHL
            case 9: this.setMem(b, Math.floor(vb / 2)); break;        // SHR (arithmetic)
            case 10: if (vb < 0) this.pc = c; break;                  // CMP (BLTZ)
            case 11: this.setMem(b, this.getMem(this.pointer(vc))); break;   // LOAD
            case 12: this.setMem(b, vb & vc); break;                  // AND
            case 13: this.setMem(b, vb | vc); break;                  // OR
            case 14: this.setMem(b, vb ^ vc); break;                  // XOR
            case 15: this.carry = this.subCarry(vb, vc); this.setMem(b, vb - vc); break;  // SUB
            case 16: {                                                // FIND
                let found = 0;
                for (let i = 0; i < this.m; i++) {
                    if (this.memory[i] === vc) { found = i; break; }
                }
                this.setMem(b, found);
                break;
            }
            case 17: this.setMem(b, vc); this.setMem(c, vb); break;  // SWAP
            case 18: if (vb < 0) this.setMem(b, vc); break;           // CMOV
            case 19: this.setMem(b, vb * 2 + (vb < 0 ? vc : 0)); break;   // MULACC
            case 20: this.setMem(this.pointer(vc), vb); break;        // STORE

            case 21: this.setMem(b, Math.max(vb, vc)); break;         // MAX
            case 22: this.setMem(b, this.logadd(vb, vc)); break;      // LOGADD
            case 23: this.setMem(b, this.getMem(this.xlatAddress(vb, c))); break;   // XLAT
            case 24: this.setMem(b, this.shift(vb, vc)); break;       // SHIFT
            case 25: {                                                // ADC
                const total = this.unsigned(vb) + this.unsigned(vc) + this.carry;
                this.carry = Math.floor(total / this.mod);
                this.setMem(b, total);
                break;
            }
            case 26: {                                                // MULSTEP on (b, b^1)
                const lo = this.unsigned(this.getMem(b ^ 1));
                const total = this.unsigned(vb) + (lo % 2 ? this.unsigned(vc) : 0);
                this.setMem(b, Math.floor(total / 2));
                this.setMem(b ^ 1, (total % 2) * this.half + Math.floor(lo / 2));
                break;
            }
            case 27: {                                                // DIVSTEP on (b, b^1)
                const quo = this.unsigned(this.getMem(b ^ 1));
                let r = 2 * this.unsigned(vb) + Math.floor(quo / this.half);
                let q = (quo * 2) % this.mod;
                if (r >= this.unsigned(vc)) { r -= this.unsigned(vc); q += 1; }
                this.setMem(b, r);
                this.setMem(b ^ 1, q);
                break;
            }
            case 28: {                                                // LFSR
                const u = this.unsigned(vb);
                this.setMem(b, Math.floor(u / 2) ^ (u % 2 ? this.unsigned(vc) : 0));
                break;
            }
            case 29: {                                                // CLZ
                const u = this.unsigned(vc);
                this.setMem(b, N - (u === 0 ? 0 : Math.floor(Math.log2(u)) + 1));
                break;
            }
            case 30: this.setMem(b, Math.abs(vc)); break;             // ABS
            case 31: {                                                // ARGMAX over 2^mem[b] words at c
                let best = 0;
                const count = 2 ** vb;
                for (let j = 1; j < count; j++) {
                    if (this.getMem(c + j) > this.getMem(c + best)) best = j;
                }
                this.setMem(b, best);
                break;
            }

            case 64: this.pc = this.cmdStart + this.unsigned(vc); break;    // JMPI
            case 65: {                                                // FALIGN (b: exponent, b^1: mantissa)
                const d = vc - vb;
                if (d > 0) {
                    this.setMem(b ^ 1, Math.floor(this.getMem(b ^ 1) / 2 ** Math.min(d, N - 1)));
                    this.setMem(b, vc);
                }
                break;
            }
            case 66: {                                                // FNORM
                const mant = this.getMem(b ^ 1);
                if (mant === 0) {
                    this.setMem(b, -this.half);
                    this.setMem(b ^ 1, 0);
                    break;
                }
                const u = this.unsigned(mant);
                const sign = Math.floor(u / this.half);
                let run = 0;
                while (run < N && Math.floor(u / 2 ** (N - 1 - run)) % 2 === sign) run++;
                const k = run - 1;
                this.setMem(b, vb - k);
                this.setMem(b ^ 1, mant * 2 ** k);
                break;
            }
            case 67: this.vlLog = vb; break;                          // VSETVL
            default:
                if (PAGE2.has(op)) { this.vector(op, b, c, vc); break; }
                throw new Error(`unknown opcode ${op}`);
        }
        return false;
    }

    vrand(seed, tag) {
        const [cs, K, h1, h2] = VRAND_CONSTANTS[this.N];
        let u = this.unsigned(seed);
        cs.forEach((cp, p) => { if (Math.floor(tag / 2 ** p) % 2) u = (u + cp) % this.mod; });
        const v = u ^ Math.floor(u / 2 ** h1);
        const w = Number((BigInt(K) * BigInt(v)) % BigInt(this.mod));
        return this.clamp(w ^ Math.floor(w / 2 ** h2));
    }

    /** One vector instruction over the aligned 2^vlLog-word blocks at b and c;
     *  every operand is read before any element is written. */
    vector(op, b, c, scalar) {
        const t = this.vlLog, vl = 2 ** t, N = this.N;
        const bb = b - (b % vl), cb = c - (c % vl);
        const old = this.memory.slice(), oldc = this.vcarry.slice();
        const get = col => (col - this.s >= 0 && col - this.s < this.m) ? old[col - this.s] : 0;
        const carryAt = col => (col - this.s >= 0 && col - this.s < this.m) ? oldc[col - this.s] : 0;
        const msk = this.unsigned(scalar) % vl;
        const bitrev = i => { let r = 0; for (let k = 0; k < t; k++) r = r * 2 + (Math.floor(i / 2 ** k) % 2); return r; };
        const trans = i => {
            if (t % 2) return i;
            const a = t / 2, lowMask = 2 ** a;
            return (i % lowMask) * lowMask + Math.floor(i / lowMask);
        };
        for (let i = 0; i < vl; i++) {
            const col = bb + i, j = col - this.s;
            if (j < 0 || j >= this.m) continue;
            const x = old[j];
            let y = 0, r;
            if ([68, 69, 70, 71, 72].includes(op)) y = get(cb + i);
            else if ([84, 85, 86, 87, 88].includes(op)) y = get(bb + (i ^ msk));
            switch (op) {
                case 68: case 84: r = y; break;                                     // VMOV, VPERMX
                case 69: case 85: r = x + y; this.vcarry[j] = this.addCarry(x, y); break;   // VADD, VADDX
                case 70: r = x - y; this.vcarry[j] = this.subCarry(x, y); break;    // VSUB
                case 71: case 87: r = Math.max(x, y); break;                        // VMAX, VMAXX
                case 72: case 88: r = this.logadd(x, y); break;                     // VLOGADD, VLOGADDX
                case 73: {                                                          // VCADD
                    const k = carryAt(cb + i);
                    r = x + k;
                    this.vcarry[j] = Math.floor((this.unsigned(x) + k) / this.mod);
                    break;
                }
                case 86: r = (col & this.unsigned(scalar)) ? y - x : x + y; break;  // VBFLY
                case 74: r = scalar; break;                                         // VFILL
                case 75: r = this.shift(x, scalar); break;                          // VSHIFT
                case 76: r = get(this.xlatAddress(x, c)); break;                    // VXLAT
                case 77: r = get(cb + bitrev(i)); break;                            // VREV
                case 78: r = get(cb + trans(i)); break;                             // VTRANS
                case 79: r = this.vrand(scalar, j); break;                          // VRAND
            }
            this.memory[j] = this.clamp(r);
        }
    }

    /** Run at most maxSteps; the returned instruction count excludes HALT. */
    run(maxSteps = 50000) {
        let steps = 0;
        while (steps < maxSteps) {
            if (this.step()) return { steps, halted: true };
            steps++;
        }
        return { steps, halted: false };
    }

    /** memIdx excludes the scratchpad address offset. */
    readVar(memIdx) {
        return this.memory[memIdx];
    }

    writeVar(memIdx, value) {
        this.memory[memIdx] = this.clamp(value);
    }

    getMemorySnapshot() {
        return this.memory.slice();
    }

    /** Read memory words from a flattened row-major bipolar state. */
    syncFromState(flat, config) {
        const n = config.n;
        const N = config.N;
        const s = config.s;
        const idxMem = config.idx_memory;
        for (let i = 0; i < this.m; i++) {
            const col = s + i;
            let value = 0;
            for (let bit = 0; bit < N; bit++) {
                if (flat[(idxMem + bit) * n + col] > 0) {
                    value |= (1 << (N - 1 - bit));
                }
            }
            if (value >= (1 << (N - 1))) value -= (1 << N);
            this.memory[i] = value;
        }
    }

    /** Replace the memory words in a flattened bipolar state. */
    syncToState(flat, config) {
        const n = config.n;
        const N = config.N;
        const s = config.s;
        const idxMem = config.idx_memory;
        for (let i = 0; i < this.m; i++) {
            const col = s + i;
            let val = this.memory[i];
            if (val < 0) val += (1 << N);
            for (let bit = 0; bit < N; bit++) {
                flat[(idxMem + bit) * n + col] = ((val >> (N - 1 - bit)) & 1) ? 1.0 : -1.0;
            }
        }
    }
}

/**
 * Create an ISA interpreter from a compile result.
 * @param {{ config, memory, commands }} compileResult - dimensions, memory and instructions
 * @returns {ISAInterpreter}
 */
function createInterpreter(compileResult) {
    const cfg = compileResult.config;
    return new ISAInterpreter(cfg.s, cfg.m, cfg.N, compileResult.memory, compileResult.commands);
}

if (typeof module !== 'undefined') module.exports = { ISAInterpreter, createInterpreter, logaddTable };
