"""
Dataset loader for chess QA pairs (image + question + answer).
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from typing import List, Dict, Optional
import json


class ChessQADataset(Dataset):
    """
    Dataset for chess question-answering.
    
    Each sample contains:
    - Image: Chess board image
    - Question: Question about the position
    - Answer: Ground truth answer
    - FEN: Optional FEN string for context
    """
    
    def __init__(
        self,
        image_paths: List[str],
        questions: List[str],
        answers: List[str],
        fens: Optional[List[str]] = None,
        image_transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to chess board images
            questions: List of question strings
            answers: List of answer strings
            fens: Optional list of FEN strings
            image_transform: Image transformation pipeline
        """
        assert len(image_paths) == len(questions) == len(answers), \
            "image_paths, questions, and answers must have same length"
        
        if fens is not None:
            assert len(fens) == len(image_paths), "fens must have same length as image_paths"
        
        self.image_paths = image_paths
        self.questions = questions
        self.answers = answers
        self.fens = fens
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.image_transform:
            image = self.image_transform(image)
        
        # Get question and answer
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Optional FEN
        fen = self.fens[idx] if self.fens is not None else None
        
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "fen": fen,
            "image_path": image_path
        }


def create_qa_dataset_from_benchmark(
    dataset_csv: str,
    images_dir: str,
    questions: List[Dict],
    num_samples: Optional[int] = None
) -> ChessQADataset:
    """
    Create QA dataset from benchmark dataset CSV.
    
    Args:
        dataset_csv: Path to CSV with image paths and FENs
        images_dir: Directory containing images
        questions: List of question dictionaries from questions.py
        num_samples: Optional limit on number of samples
        
    Returns:
        ChessQADataset
    """
    # Load dataset CSV
    df = pd.read_csv(dataset_csv)
    
    # Filter to available images
    image_paths = []
    fens = []
    
    for _, row in df.iterrows():
        image_path = row.get('image_path', '')
        fen = row.get('fen', '')
        
        # Try multiple path formats
        if os.path.isabs(image_path):
            full_path = image_path
        else:
            # Try relative to images_dir
            filename = os.path.basename(image_path)
            full_path = os.path.join(images_dir, filename)
        
        if os.path.exists(full_path) and fen:
            image_paths.append(full_path)
            fens.append(fen)
    
    if num_samples:
        image_paths = image_paths[:num_samples]
        fens = fens[:num_samples]
    
    # Create QA pairs: for each image, create samples for each question
    all_image_paths = []
    all_questions = []
    all_answers = []
    all_fens = []
    
    # We'll need to generate answers from FEN using ground truth extractor
    # For now, we'll create the structure and answers can be generated during training
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from benchmarking.ground_truth import GroundTruthExtractor
        # Pass dataset_csv so it can use best_continuation field instead of API
        gt_extractor = GroundTruthExtractor(dataset_csv=dataset_csv)
        
        for image_path, fen in zip(image_paths, fens):
            for question in questions:
                all_image_paths.append(image_path)
                all_questions.append(question['prompt'])
                all_fens.append(fen)
                
                # Generate ground truth answer
                qtype = question['type']
                try:
                    if qtype == "fen_extraction":
                        answer = fen
                    elif qtype == "piece_count":
                        answer = str(gt_extractor.get_piece_count(fen))
                    elif qtype == "check_status":
                        answer = str(gt_extractor.get_check_status(fen))
                    elif qtype == "material_balance":
                        answer = str(gt_extractor.get_material_balance(fen))
                    elif qtype == "material_advantage":
                        answer = str(gt_extractor.get_material_advantage(fen))
                    elif qtype == "material_count_white":
                        answer = str(gt_extractor.get_material_count(fen, color="white"))
                    elif qtype == "material_count_black":
                        answer = str(gt_extractor.get_material_count(fen, color="black"))
                    elif qtype == "queen_count":
                        answer = str(gt_extractor.get_queen_count(fen))
                    elif qtype == "minor_piece_balance":
                        answer = str(gt_extractor.get_minor_piece_balance(fen))
                    elif qtype == "rook_count":
                        answer = str(gt_extractor.get_rook_count(fen))
                    elif qtype == "pawn_advantage":
                        answer = str(gt_extractor.get_pawn_advantage(fen))
                    elif qtype == "best_move":
                        answer = str(gt_extractor.get_best_move(fen))
                    elif qtype == "tactical_pattern":
                        answer = str(gt_extractor.get_tactical_patterns(fen))
                    elif qtype == "castling_available":
                        answer = str(gt_extractor.get_castling_rights(fen))
                    elif qtype == "piece_on_square":
                        answer = str(gt_extractor.get_piece_on_square(fen, "e4"))
                    else:
                        answer = "N/A"
                except Exception as e:
                    print(f"Warning: Could not generate answer for {qtype}: {e}")
                    answer = "N/A"
                
                all_answers.append(answer)
    
    except ImportError:
        print("Warning: Could not import GroundTruthExtractor, using placeholder answers")
        for image_path, fen in zip(image_paths, fens):
            for question in questions:
                all_image_paths.append(image_path)
                all_questions.append(question['prompt'])
                all_fens.append(fen)
                all_answers.append("Placeholder answer")
    
    # Image transform
    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return ChessQADataset(
        image_paths=all_image_paths,
        questions=all_questions,
        answers=all_answers,
        fens=all_fens,
        image_transform=image_transform
    )


def load_qa_dataset_from_json(json_path: str, image_transform=None) -> ChessQADataset:
    """
    Load QA dataset from JSON file.
    
    JSON format:
    [
        {
            "image_path": "path/to/image.png",
            "question": "What is the FEN?",
            "answer": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        },
        ...
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_paths = [item['image_path'] for item in data]
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    fens = [item.get('fen') for item in data]
    
    if image_transform is None:
        from torchvision import transforms
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return ChessQADataset(
        image_paths=image_paths,
        questions=questions,
        answers=answers,
        fens=fens,
        image_transform=image_transform
    )

