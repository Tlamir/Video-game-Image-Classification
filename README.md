# Video-game-Image-Classification

A deep learning project implementing CNN (Convolutional Neural Network) architecture for automatic classification of video game screenshots. This project demonstrates comprehensive deep learning techniques including data preprocessing, model development, evaluation, and interpretation.

## Links
#### Kaggle: https://www.kaggle.com/code/yiithangven/yigithan-guven-videogame-image-classification
#### Dataset: https://www.kaggle.com/datasets/juanmartinzabala/videogame-image-classification

## Project Scope

This project addresses the challenge of automatically identifying video games from screenshots using computer vision and deep learning techniques. The system analyzes visual elements such as user interfaces, art styles, game environments, and distinctive graphical features to classify screenshots into their respective game categories.

## Technical Implementation

### CNN Architecture
- **Model Type**: Custom Convolutional Neural Network
- **Input Dimensions**: 192x192x3 RGB images
- **Architecture Design**: Progressive feature extraction with increasing filter complexity
- **Total Parameters**: ~1.2 million (optimized for efficiency)
- **Output**: 21-class classification with softmax activation

### Core Components
- **Convolutional Layers**: Feature extraction with ReLU activation
- **Pooling Layers**: Spatial dimension reduction and translation invariance
- **Dropout Layers**: Regularization to prevent overfitting (rates: 0.25, 0.3, 0.5)
- **Dense Layers**: Fully connected layers for final classification
- **Batch Normalization**: Training stability and convergence acceleration

### Data Processing Pipeline
- **Data Augmentation**: RandomFlip, RandomRotation, RandomZoom for generalization
- **Normalization**: Pixel value scaling to [0,1] range
- **Train-Validation-Test Split**: 80%-20% split with separate test set
- **Memory Optimization**: Efficient data loading without caching

## Dataset

- **21 Video Game Classes**: Including popular titles like CS:GO, Fortnite, FIFA21, Minecraft, Valorant, and others
- **115,000+ Screenshots**: Large-scale dataset ensuring robust training
- **High Quality Images**: Diverse gameplay scenarios and visual conditions

### Supported Games
ApexLegends • CSGO • ClashRoyale • DeathByDaylight • Dota2 • EscapeFromTarkov • FIFA21 • Fortnite • FreeFire • GTAV • LeagueOfLegends • Minecraft • Overwatch • PUBG_Battlegrounds • Rainbows • RocketLeague • Rust • SeaOfThieves • Valorant • Warzone • WorldOfWarcraft

## Model Evaluation & Analysis

### Performance Metrics
- **Training Accuracy**: 89.4%
- **Validation Accuracy**: 87.3%
- **Test Accuracy**: 85.8%
- **Training Epochs**: 18 epochs with early stopping capability

### Evaluation Methods
- **Accuracy & Loss Curves**: Epoch-by-epoch training progress visualization
- **Confusion Matrix**: Detailed classification performance analysis
- **Classification Report**: Precision, recall, and F1-score for each game class
- **Model Interpretability**: Analysis of prediction patterns and confidence distributions

## Advanced Techniques

### Transfer Learning (Bonus Implementation)
- **Base Model**: MobileNetV2 pre-trained on ImageNet
- **Fine-tuning Strategy**: Frozen base layers with custom classification head
- **Comparative Analysis**: Performance comparison between custom CNN and transfer learning approach
- **Results**: Transfer learning achieved faster convergence with competitive accuracy

### Model Interpretability
- **Prediction Analysis**: Confidence pattern analysis across different games
- **Error Analysis**: Identification of commonly confused game pairs
- **Feature Importance**: Discussion of visual elements the model focuses on
- **Architecture Analysis**: Layer-by-layer capacity and parameter distribution

## Key Features

- **High Accuracy**: Consistently identifies games with 87%+ accuracy
- **Fast Inference**: Optimized for real-time classification
- **Robust Design**: Handles various screenshot qualities and conditions
- **Memory Efficient**: Lightweight model suitable for deployment
- **Well Regularized**: Prevents overfitting through multiple techniques

## Model Architecture

Progressive Convolutional Neural Network with:
- 4 convolutional blocks with increasing complexity (32→64→128→256 filters)
- Batch normalization for training stability
- Dropout layers for regularization
- Global average pooling for parameter reduction
- Dense classification layers with softmax output

## Model Capabilities

The trained model can process any video game screenshot and return:
- Predicted game title from the 21 supported classes
- Confidence score indicating prediction certainty
- Real-time classification suitable for streaming applications

## Technical Highlights

### Data Processing
- Normalized input images (192x192 pixels)
- Data augmentation for improved generalization
- Balanced dataset across all game classes

### Training Strategy
- Progressive learning with learning rate scheduling
- Multiple regularization techniques to prevent overfitting
- Validation-based early stopping and model checkpointing

### Model Optimization
- Memory-efficient architecture design
- Optimized for single GPU training
- Fast inference suitable for real-time applications

## Applications

- **Game Streaming**: Automatic game detection for streaming platforms
- **Content Moderation**: Identify game content in user uploads
- **Analytics**: Track gaming trends and popular titles
- **Accessibility**: Help visually impaired users identify games
- **Research**: Dataset for computer vision and gaming studies
