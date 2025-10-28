from util import mkdir


# directory to store the results
results_dir = '/mntnfs/med_data5/wanyue/real-vs-fake/results/test_classification'
#mkdir(results_dir)

# root to the testsets
dataroot = '/mntnfs/med_data5/wanyue/real-vs-fake/results/LNP/test'

# list of synthesis algorithms
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
vals = ['stylegan']
# indicates if corresponding testset has multiple classes
multiclass = [0]

# model
model_path = '/home/wanyue/CNNDetection-master/checkpoints/LNPPP/model_epoch_best.pth'
