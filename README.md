# Reconstructing Facial Sketches from Voices using Dictionary Learning 

This repository includes the code to reconstruct facial sketches from voices. 

### Repository Structure

- `mlsp_project_code` contains the old code from the course project 
- `voice2face` contains the refactored code 
- `speaker_id_weights.pth` is the speaker ID weights file

### How to install

Use the `requirements.txt` file to install the dependencies. Preferably, use conda since PyTorch likes conda. 

### How to run 

The dataloader expects a file containing the filepath of each sample along with labels. For example, take a look at `vgg_voxceleb_edge_preserving.txt`. Then train the model as follows:

```bash
$ python train.py vgg_voxceleb_edge_preserving.txt
```

### How to contribute 

1. Make sure you can understand and can run the code inside `voice2face`
2. Create a new directory in this repo, call it `face_id` or `speaker_id` and use the code inside `voice2face` as a template (makes it easier for everyone to read your code)
3. I used Weights&Biases to visualize the training metrics, feel free to use it, take it out or use anything else you like 


### Questions 

Please reach out to me (Mahmoud) on Slack




