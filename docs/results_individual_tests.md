admin@DESKTOP-MFTG7IE MINGW64 /d/vivek_projects/vichar/vichar-clip/Notebooks (main)
$ python benchmark_individual.py checkpoints/large_1000/fen_only_model/clip_chess_epoch_5.pt large_datasets/fen_only
2025-08-21 18:44:34,134 - INFO - Using device: cuda
2025-08-21 18:44:35,728 - INFO - Successfully loaded model weights from: checkpoints/large_1000/fen_only_model/clip_chess_epoch_5.pt
2025-08-21 18:44:36,215 - INFO - Dataset loaded: 61169 examples
Encoding test data: 100%|████████████████████| 956/956 [01:15<00:00, 12.59it/s]

📊 Evaluation Results for checkpoints/large_1000/fen_only_model/clip_chess_epoch_5.pt
============================================================
Image-to-Text Top-1 Acc (%): 16.65%
Image-to-Text Top-5 Acc (%): 48.76%
Text-to-Image Top-1 Acc (%): 20.30%
Text-to-Image Top-5 Acc (%): 55.90%
Total Samples: 61169
============================================================
(pytorch_5070ti)
admin@DESKTOP-MFTG7IE MINGW64 /d/vivek_projects/vichar/vichar-clip/Notebooks (main)
$ppython benchmark_individual.py checkpoints/large_1000/fen_move_model/clip_chess_epoch_5.pt large_datasets/fen_move
2025-08-21 18:47:32,593 - INFO - Using device: cuda
2025-08-21 18:47:34,202 - INFO - Successfully loaded model weights from: checkpoints/large_1000/fen_move_model/clip_chess_epoch_5.pt
2025-08-21 18:47:34,709 - INFO - Dataset loaded: 61169 examples
Encoding test data: 100%|█████████████████████████████████████████████████| 956/956 [01:23<00:00, 11.43it/s]

📊 Evaluation Results for checkpoints/large_1000/fen_move_model/clip_chess_epoch_5.pt
============================================================
Image-to-Text Top-1 Acc (%): 12.52%
Image-to-Text Top-5 Acc (%): 40.87%
Text-to-Image Top-1 Acc (%): 12.58%
Text-to-Image Top-5 Acc (%): 41.28%
Total Samples: 61169
============================================================
(pytorch_5070ti)
