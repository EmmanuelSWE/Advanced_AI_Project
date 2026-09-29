import utils.meta_wrld_utils as metaUtils
import utils.csv_utils as csvUtils
import utils.image_utils as imgUtils



#metaUtils.runAndStoreDemonstrations()


# laod the image
imageTest = imgUtils.loadImageFromPath('../DATASET_generations/drawer-open-v3/drawer-open-v3_step0_id2.png')


print(imgUtils.getSize(imageTest))
print(imageTest.mode)