# Motion Estimation - Block Matching Solutions

## full_search/cuda_uncoalesced_optimized

Questa soluzione usa un unico kernel `fullSearchKernel` che calcola SAD e riduce per il miglior MV nello stesso kernel. Sfrutta una griglia 2D `(blocksX, blocksY)` con block 3D `(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ)` per parallelizzare il calcolo della SAD lungo la dimensione Z. Gli accessi a global memory non sono coalesced perché ogni thread (X, Y) dello stesso blocco CUDA accede a blocchi del reference frame differenti. Supporta sia full_search che range search.

### Scelta dei thread e dimensione della SMEM

Lato host, il block 3D è determinato in modo adattivo dai vincoli della GPU. Se `threadsPerBlockX * threadsPerBlockY <= maxThreadsPerBlock`, si calcola `threadsPerBlockZ` come il massimo consentito (fino a 64). Se invece le dimensioni X, Y sono troppo grandi, si ripiegha su un valore fisso `(16, 16, 4)`. La dimensione della SMEM è calcolata per ospitare il blocco corrente, i SAD parziali da ridurre, e i buffer di riduzione finali.

### `fullSearchKernel`

Kernel unico che calcola tutte le SAD e trova il miglior MV per ogni blocco del frame corrente.

#### Grid, block, SMEM e global memory

- **Grid**: 2D `(blocksX, blocksY)`. Ogni blocco CUDA processa un singolo blocco del frame corrente.
- **Block**: 3D `(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ)`. Sfrutta la dimensione Z per parallelizzare il calcolo della SAD sui pixel.
- **SMEM**: Dimensione totale `blockSize² * sizeof(unsigned char) + threadsPerBlock * sizeof(int) + 4 * (threadsPerBlockX * threadsPerBlockY) * sizeof(int)`:
  - `s_curr`: blocco del frame corrente (`blockSize²` byte)
  - `shared_block_thread_sad_partial`: array SAD parziali da ridurre lungo Z (`threadsPerBlock` int)
  - `shared_block_thread_sad`: array SAD finale per ogni thread (X, Y) (`threadsPerBlockX * threadsPerBlockY` int)
  - `shared_block_thread_dx`, `shared_block_thread_dy`, `shared_block_thread_dist`: array coordinate e distanze per tie-breaking (3 × `threadsPerBlockX * threadsPerBlockY` int)
- **Global memory**:
  - **Input**: `d_curr` e `d_ref`, frame corrente e riferimento
  - **Output**: `d_mv`, `blocksX * blocksY` MotionVector

#### Esecuzione

1. **Load SMEM**: Tutti i thread caricano il blocco del frame corrente in `s_curr` con stride `threadsPerBlock`. 
2. **Position Processing**: Se `bounds.totalPositions <= threadsPerBlock`, ogni thread (X, Y) processa al massimo una posizione e thread lungo Z parallelizzano il calcolo della SAD. Se più posizioni, ogni thread (X, Y) processa più posizioni in sequenza con SAD parallelizzato lungo Z.

3. **Partial SAD Computation**: La funzione `computeSAD_device_partial()` calcola la SAD per un subset di pixel (determinato da `threadIdx.z` e `pixelsPerThread`). I risultati vengono scritti in `shared_block_thread_sad_partial[tidx_3d]`. 

4. **Reduction along Z**: Loop con stride di potenze di 2 riduce i SAD parziali lungo Z, ovviamente, ad ogni step è necessario la `__syncthreads()`.
Dopo la riduzione, thread con `threadIdx.z == 0` hanno il SAD totale in `shared_block_thread_sad_partial[tidx_3d]`. Questo valore viene copiato in `shared_block_thread_sad[tidx_2d]` insieme a dx, dy e distanza.

5. **Final Reduction**: Thread `(0, 0, 0)` esegue una riduzione sequenziale su tutti i SAD calcolati dai thread (X, Y, 0) per trovare il miglior MV con tie-breaking per distanza.

6. **Write GMEM**: Thread `(0, 0, 0)` scrive il MV finale in `d_mv` (global memory).

## full_search/cuda_optimized

Questa soluzione usa due kernel: `computeSADKernel` e `findBestMVKernel`. Il primo calcola tutte le SAD con grid 3d (`blocksX × blocksY × maxCandidates`), il secondo riduce per trovare il miglior MV. 
Questa separazione è scelta perché un kernel unico avrebbe un grid 2D piccola, e le riduzioni per scegliere il migliore MV non sono associative,m quindi sono difficili da parallelizzare. La soluzione sfrutta shared memory per caricamenti coalesced e riduzioni in SMEM. Supporta sia full_search che range search.


### Scelta dei thread per blocco e dimensione della SMEM

Lato host, sia il numero di thread per blocco cuda che la dimensione della SMEM, sono determinati da trovare una configurazione adatta che soddisfi i vincoli della GPU. Se non è possibile, si ripiegha su 32 thread per blocco cuda, la dimensione della SMEM è cacolata di conseguenza. 

### `computeSADKernel`

Calcola la SAD tra il blocco nel current frame e ogni posizione candidata nel reference frame.

#### Grid, block, SMEM e global memory

- **Grid**: 3D `(blocksX, blocksY, maxCandidates)`. Ogni blocco CUDA processa un singolo blocco del frame corrente ( `x`, `y`) contro una singola posizione candidata nel reference frame (`ref_x`, `ref_y`).
- **Block**: Unidimensionale, `threadsPerBlock` thread.
- **SMEM**: Dimensione totale `smKSad = 2 * pixelsPerBlock * sizeof(unsigned char) + threadsPerBlock * sizeof(int)`:
  - `s_curr`: blocco del frame corrente (`pixelsPerBlock` byte)
  - `s_ref`: blocco del frame di riferimento (`pixelsPerBlock` byte)
  - `s_partial_sad`: array di risultati parziali SAD, uno per thread (`threadsPerBlock * sizeof(int)`)
- **Global memory**:
  - **Input**: `d_curr` e `d_ref`.
  - **Output**: `d_sad` risultati SAD per tutte le posizioni candidate, che verranno letti dal kernel successivo (`blocksX * blocksY` MotionVector)

#### Esecuzione

1. **Write SMEM**: Ogni thread carica più pixel con stride `blockDim.x` in `s_curr` e `s_ref`. I caricamenti sono coalesced perché i thread accedono a posizioni contigue in global memory.
2. **Partial SAD computation**: Ogni thread computa la SAD per la porzione di pixel assegnata (calcolata come `pixelsPerThread = (pixelsPerBlock + blockDim.x - 1) / blockDim.x`) e scrive il risultato in `s_partial_sad[threadIdx.x]`. 

3. **Reduction in SMEM**:
In ogni iterazione, la prima metà attiva dei thread somma i valori dalla seconda metà (in `s_partial_sad`), riducendo i risultati parziali fino a ottenere un unico valore in `s_partial_sad[0]`. Ovviamente ad ogni step è necessario sicronizzare. 
4. **Write GMEM**: Il thread 0 scrive il SAD finale in `d_sad` (global memory).

### `findBestMVKernel`

Identifica il motion vector migliore per ogni blocco del frame corrente, utilizzando i risultati SAD calcolati dal kernel precedente.

#### Grid, block, SMEM e global memory

- **Grid**: 2D `(blocksX, blocksY)`. Ogni blocco CUDA determina il miglior MV per un blocco del frame corrente.
- **Block**:  `threadsPerBlock` thread.
- **SMEM**: Dimensione totale `smKBestMV = 4 * threadsPerBlock * sizeof(int)`:
  - `s_sad`: array SAD per ogni thread.
  - `s_dx`: array motion vector X per ogni thread.
  - `s_dy`: array motion vector Y per ogni thread.
  - `s_dist`: array distanze  per tie-breaking.
- **Global memory**:
  - **Input**: `d_sad`, i risultati del kernel precedente (`blocksX * blocksY * maxCandidates` int)
  - **Output**: `d_mv`, il motion vector finale per ogni blocco del frame corrente (`blocksX * blocksY` MV)

#### Esecuzione

1. **Local Best Search**: Ogni thread itera su un subset di posizioni candidate (stride `blockDim.x`), calcolando localmente il miglior SAD e il vettore di movimento associato. I valori locali vengono salvati in (SMEM) `s_sad[threadIdx.x]`, `s_dx[threadIdx.x]`, `s_dy[threadIdx.x]`, `s_dist[threadIdx.x]`.

2. **Sequential Reduction**: Il thread 0 esegue una riduzione sequenziale degli array in SMEM per trovare il migliore globale. La riduzione è sequenziale (non parallela) perché il confronto con tie-breaking per distanza non è associativo.
3. **Write GMEM**: Il thread 0 scrive il MV finale in `d_mv` (global memory).

