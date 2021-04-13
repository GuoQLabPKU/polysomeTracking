from polysome import Polysome
import numpy as np

polysome1 = Polysome(input_star = 'simOrderRandomized.star')
polysome1.classify['clustThr'] = 5
polysome1.classify['relinkWithoutSmallClasses'] = 0
polysome1.avg['minNumTransform'] = np.inf
polysome1.sel[0]['minNumTransform'] = 0

polysome1.creatOutputFolder()
polysome1.calcTransForms()
polysome1.groupTransForms()
transListSel, selFolds = polysome1.selectTransFormClasses()
polysome1.alignTransforms()
polysome1.find_connectedTransforms()
polysome1.analyseTransFromPopulation()
polysome1.find_transFromNeighbours()