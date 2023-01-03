from PIL import Image
###
## Importing description text files
###
with open('dataset_topic.txt') as f:
    dataset_topic = f.read()
    f.close()
with open('dataset_origin.txt') as f:
    dataset_origin = f.read()
    f.close()
with open('pca_description.txt') as f:
    pca_desc = f.read()
    f.close()
with open('lda_desc.txt') as f:
    lda_desc = f.read()
    f.close()
with open('iso_desc.txt') as f:
    iso_desc = f.read()
    f.close()
with open('iso_desc.txt') as f:
    decision_tree_desc = f.read()
    f.close()
with open('adaboost_desc.txt') as f:
    adaboost_desc = f.read()
    f.close()
with open('gaussian_NB_desc.txt') as f:
    gaussian_NB_desc = f.read()
    f.close()
with open('target_desc.txt') as f:
    target_desc = f.read()
    f.close()
with open('down_sampling_algorithm.txt') as f:
    down_sampling_algorithm = f.read()
    f.close()
with open('sleep_mental_corr.txt') as f:
    sleep_mental_corr = f.read()
    f.close()
with open('correlations.txt') as f:
    correlations = f.read()
    f.close()
decision_tree_image = Image.open('Decision_tree_pic.png')