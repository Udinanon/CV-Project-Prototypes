# CV Project Prototypes
## Prototypes and approaches

### CNN
the fundamental idea is to use a CNN as an end-to-end method
there's tons of working models from the internet, we can 100% do at least a few versions using these and some transfer learning to just get some results

should be the best method, if not for simplicity for performance

#### Datasets
Hands Over Face https://drive.google.com/file/d/1hHUvINGICvOGcaDgA5zMbzAIUv7ewDd3/edit
EgoHands http://vision.soic.indiana.edu/projects/egohands/

Oxford Hands https://www.robots.ox.ac.uk/~vgg/data/hands/


#### TODO:
 - [x] setup goole colab   
   - [x] load dataset on google colab
 - [x] setup pytorch 
   - [x] add pytorch to googlecolab
   - [x] load models in pytorch for image segmentation 
   - [x] run them on small sample
 - [ ] use pytorch segmentation models
   - [ ] learn the models design enought to do some basic retraining
   - [ ] retrain on dataset
   - [ ] check performance 
 - read papers and sources to find other models

#### Material
https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation
pytorch CNN segmentation models, could be great candidates for some form of transfer learning
3 models in total, 2015, 2017, 2019

we could pick one or two of these, although wome work will be needed to understand how to retrain them exactly


https://towardsdatascience.com/train-an-object-detector-using-tensorflow-2-object-detection-api-in-2021-a4fed450d1b9
useful tutorial although very basic


https://learnopencv.com/mask-r-cnn-instance-segmentation-with-pytorch/

all pytorch models, both semantic and instance segmentators
https://pytorch.org/vision/stable/models.html#
Mask RCNN seems easy and powerful

https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#semantic-segmentation-models
for result visualization

### Simple patter matching
Bruno's idea
simple, might be a good naive approach

use color thresholding to detect skin
then use moving window to make simple patter matching on shapes of hands from ground truth data
check with threshold

probably vulnerable to weird scales, some occlusion and rotation 
needs approach for segmentation beyond detection

#### TODO:
 - [ ] get a clear pseudocode/logical steps of algorithm    
   - [ ] search it online
 - [ ] define basic hand images set
 - [ ] do basic implementation in python

### Bag of Words
my idea kinda
i want to see if a bag of words again from a ground truth can be used decently 

use color thresholding to detect skin
then run bag of words method 
check with threshold

vulnearble to weird shapes, occlusion
needs approach for segmentation beyond detection
#### TODO:
 - [x] search "bag of words opencv python"
 - [ ] implement basic version and understand fundamentals of method
 - [ ] research

### Cascade ?
There is a full cascade trainer in OpenCV
issue is that the trainer is already implemented, we could just train it on the data and use the resulting code but idk if it's valid

would probably be quite resilient to many problems and issues

https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html 
https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html


#### TODO:
 - [x] discuss with colleagues
 - [ ] understand if we could use the internal trianer and then just write the model code
 - [ ] try and see if it works

## other 

### references and bi0bliographical research
i want to have quite a  bit fo references in the report
even some mediocre ones, but i like the idea of showing we did research on this

a csv/excel with all the relevant articles we read and notes on which and how they might be useful

#### TODO:
 - [ ] build small csv file tracking articles and info sources
 - [ ] do more research and steal ideas and approaches

### report
i don't think i like word but it's not a major factor for now
i'll take my notes for my protypes under markdown here and i'll show it to the others later