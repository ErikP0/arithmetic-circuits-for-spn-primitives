## Data format in results.csv

Columns
| Column            |                                                                                                |
|-------------------|------------------------------------------------------------------------------------------------|
| primitive         | the benchmarked circuit                                                                        |
| n                 | the SIMD parameter (i.e., how many circuits were executed in parallel)                         |
| offline_time      | the time in seconds of the offline/pre-processing phase                                        |
| online_time       | the time in seconds of the online phase                                                        |
| combined_time     | the time in seconds of a combined execution, i.e., offline and online phase                    |
| online_data       | the data in MB sent per player during the online phase                                         |
| combined_data     | the data in MB sent per player during the combined execution, i.e., offline and online phase   |
| offline_data      | the data in MB sent per player during the offline phase                                        |
| avg_offline_time  | the time in seconds of the offline phase per circuit (i.e., offline_time/n)                    |
| avg_online_time   | the time in seconds of the online phase per circuit (i.e., online_time/n)                      |
| avg_combined_time | the time in seconds of the combined execution per cirucit (i.e., combined_time/n)              |
| avg_online_data   | the data in MB sent per player during the online phase per circuit (i.e., online_data/n)       |
| avg_offline_data  | the data in MB sent per player during the offline phase per circuit (i.e., online_data/n)      |
| avg_combined_data | the data in MB sent per player during the combined execution per circuit (i.e., online_data/n) |


## About combined vs. offline + online
A combined execution in MP-SPDZ does not necessarily finish the offline phase before starting the online phase. Some rounds of the online phase may already execute as soon as sufficient pre-processed triples are generated.
Therefore, the execution time of the combined execution may be lower than the sum of offline and online execution time.
See also [benchmarking online phase](https://mp-spdz.readthedocs.io/en/latest/readme.html#online-only-benchmarking) and [benchmarking offline phase](https://mp-spdz.readthedocs.io/en/latest/readme.html#benchmarking-offline-phases) in MP-SPDZ.
