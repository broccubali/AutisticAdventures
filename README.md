# AutisticAdventures

```
bash
[pipeline] = ccs | cpac | dparsf | niak 
[strategy] = filt_global | filt_noglobal | nofilt_global | nofilt_noglobal
[file identifier] = the FILE_ID value from the summary spreadsheet
[derivative] = alff | degree_binarize | degree_weighted | dual_regression | ... 
               eigenvector_binarize | eigenvector_weighted | falff | func_mask | ... 
               func_mean | func_preproc | lfcd | reho | rois_aal | rois_cc200 | ... 
               rois_cc400 | rois_dosenbach160 | rois_ez | rois_ho | rois_tt | vmhc
[ext] = 1D | nii.gz
```

So
- Pipeline- cpac (takes care of preprocessing, a lot of people have used it)
- Strategy- nofilt_noglobal (raw, no averaging fmri over full brain, no noise removal)
- derivative-
    - everything with ROI
    - This is cuz everything with roi gets saved as 1d file
    - 1d file is correlation matrix of regions of interest
    - still have to make a hypergraph from this
    - still yay hypergraph
- 1d is just the processed .nii
- use nibabel with nii data (works good)
```
python
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
test_load = nib.load('/content/CMU_a_0050642_degree_binarize.nii.gz').get_fdata()
test_load.shape
for i in range(60):
  test = test_load[:,:,60-i]
  plt.imshow(test)
  plt.show()
```

Gotta figure out: 
- why the big summary thing
- who are they
