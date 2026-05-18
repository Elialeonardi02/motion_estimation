🇬🇧 English version: [README.md](README.md)
# Motion Estimation - Block Matching Solutions

## full_search/cuda_optimized

Questa ottimizzazione per CUDA utilizza la SMEM come nella versione precedente, ma con un layout di memoria più efficente per andare a migliorare gli accessi coalesced alla GMEM. 
L'implementazione è composta da due kernel principali: `computeSADKernel` e `findBestMVKernel`:
### `computeSADKernel`
Tale kernel si occupa di calcolare la SAD tra il blocco nel current frame e ogni posizione candidata nell reference frame. 
#### Struttura grid block e SMEM
Il grid associato a tale kernel è composto da 3 dimensioni: (numBlocksX, numBlocksY, numSearchPositions):
- `BlocksX`: numero di blocchi in orizzontale nel current frame. 
- `BlocksY`: numero di blocchi in verticale nel current frame.
- `maxCandidates`: numero totale di posizioni candidate nel search window.

Ed il block all'interno del grid ha un unica dimensione `threadsPerBlock` la quale si cerca di mantenere il più alta possibile, fino al massimo consentito dalla GPU.
I blocchi che compongono il grid utilizzano la SMEM con una
 dimensione di memoria pari a un totale di `smKSad = 2 * pixelsPerBlock * sizeof(char) + threadsPerBlock * sizeof(int)`:
-Il blocco corrente e quello del reference frame (`2 *pixelsPerBlock *sizeof(char)`)
-I risultati parziali della sad (`threadsPerBlock * sizeof(int)`).
#### Esecuzione
Come nella versione precedente, ogni blocco del grid elabora un singolo blocco del frame corrente: le dimensioni `BlocksX` e `BlocksY` individuano il blocco del frame corrente da processare, mentre la terza dimensione `maxCandidates` (o `numSearchPositions`) corrisponde alle diverse posizioni candidate nel frame di riferimento che devono essere valutate per quel blocco.
 
Nel codice CUDA questa mappatura è realizzata lanciando il kernel con una griglia 3D `gridKSad(blocksX, blocksY, maxCandidates)`: le coordinate `blockIdx.x` e `blockIdx.y` selezionano il blocco del frame corrente (colonna e riga), mentre `blockIdx.z` seleziona una singola posizione candidata nel frame di riferimento. Di conseguenza, ogni "fila" sull'asse Z confronta lo stesso blocco del current frame con una singola posizione candidata del reference frame, ripetendo il confronto per tutte le posizioni candidate.

I singoli thread all'interno del blocco calcolano la porzione di SAD per i pixel assegnati `pixelsperThread`, e memorizzando i risultati parziali della Sad nella SMEM. Al termine del calcolo, un thread (ad esempio `threadIdx.x == 0`) si occupa di sommare i risultati parziali e memorizzare la SAD totale per quella posizione candidata in GMEM.