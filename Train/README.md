Things to note before traning
-------------------------

There should be 3 folders, **train, val, test**, under the root directory of which the training script runs. <br/>
Train: for actually finetuning Yolo model. <br/>
Val: for evaluating the model's performance after tuning. <br/>
Test: for testing the model's function. (Optional) <br/>

There should also be 2 folders, **images, labels**, in each of the 3 folders **train, val, test**. <br/>
Images: to hold the actual training images. <br/>
Labels: to hold the annotation txt files. <br/>

For every image, there should be a annotation .txt file with the same name. <br/>
E.g. If there are 1000 .png files in **/image**, there should also be 1000 .txt files in **/labels**. <br/>

### Folder structure

                  - train  - images
                 |        |
                 |         - labels
                 |         
trainingn root   -  val    - images
                 |        |
                 |         - labels         
                 |
                  - test   - images
                          |
                           - labels

If any changes to path are made, update the corresponding part in data.yaml.
