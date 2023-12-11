# Zero-shot object detection with CLIP and TensorFlow.js

The app loads a TensorFlow.js CLIP model from [modelzoo.js](https://github.com/zemlyansky/modelzoo) and [h5 weights](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) from HF. CLIP encodes images and texts into a 512-dimensional embedding space. The app captures the webcam stream and encodes it into a 512-dimensional vector. Then it compares the vector with the embeddings of provided classes and if the target class is found, sound is played and the class is displayed.
