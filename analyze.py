from pathlib import Path
from PIL import Image
from typing import Union, List, Optional
import numpy as np

from minicpm_helper import init_model


class AnalyzeImage:
    """
    A wrapper class for MiniCPMV model that provides a simple interface for image analysis.
    """
    
    def __init__(
        self, 
        model_dir: Union[str, Path] = '../models/minicpm_v_2_6',
        llm_model_dir: str = 'language_model_int4',
        device: str = 'CPU'
    ):
        """
        Initialize the AnalyzeImage class.

        Args:
            model_dir: Path to the model directory
            llm_model_dir: Name of the language model directory
            device: Device to run inference on ('CPU' or 'GPU')
        """
        self.model_dir = Path(model_dir)
        self.ov_model = None
        self.tokenizer = None
        self._initialize_model(llm_model_dir, device)

    def _initialize_model(self, llm_model_dir: str, device: str) -> None:
        """Initialize the OpenVINO model and tokenizer."""
        self.ov_model = init_model(self.model_dir, llm_model_dir, device)
        self.tokenizer = self.ov_model.processor.tokenizer

    def analyze(
        self, 
        image: Union[Image.Image, List[Image.Image]], 
        question: str,
        stream: bool = False,
        max_new_tokens: int = 1000,
        sampling: bool = False,
        **kwargs
    ) -> Union[str, iter]:
        """
        Analyze an image or list of images with a given question.

        Args:
            image: A single PIL Image or list of PIL Images
            question: Question about the image(s)
            stream: Whether to stream the output token by token
            max_new_tokens: Maximum number of tokens to generate
            sampling: Whether to use sampling for text generation
            **kwargs: Additional arguments to pass to the model's chat method

        Returns:
            If stream=False: A string containing the model's response
            If stream=True: An iterator that yields response tokens
        """
        if self.ov_model is None:
            raise RuntimeError("Model not initialized. Please check if the model was loaded correctly.")

        # Handle single image case
        if not isinstance(image, list):
            image = [image]

        # Prepare messages - combine image and question in content
        msgs = [{"role": "user", "content": [image[0], question]}]  # For single image case

        # Call the model's chat method directly
        response = self.ov_model.chat(
            image=None,  # Image is already in msgs
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=sampling,
            stream=stream,
            max_new_tokens=max_new_tokens,
            max_slice_nums=2,
            use_image_id=False,
            **kwargs
        )

        if stream:
            # Return a generator that filters out special tokens
            def response_generator():
                for text in response:
                    # Remove special tokens if present
                    for term in self.ov_model.terminators:
                        text = text.replace(term, "")
                    yield text
            return response_generator()
        
        return response

    def __call__(
        self, 
        image: Union[Image.Image, List[Image.Image]], 
        question: str,
        **kwargs
    ) -> str:
        """
        Convenience method to call analyze() directly on class instance.
        
        Args:
            image: A single PIL Image or list of PIL Images
            question: Question about the image(s)
            **kwargs: Additional arguments to pass to analyze()
            
        Returns:
            String containing the model's response
        """
        return self.analyze(image, question, **kwargs) 