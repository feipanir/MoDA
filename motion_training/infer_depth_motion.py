import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib.data_iterators import SeqReader
from lib.img_processing import ImageProcessor, concat_depth_img
from lib.utils import get_model_opt

from evaluators import EVALUATORS
from options import InferOptions

import pdb
def normalize_trans(x):
    """Rescale translation 

    if all values are positive, rescale the max to 1.0
    otherwise, make sure the zeros be mapped to 0.5, and
    either the max mapped to 1.0 or the min mapped to 0
    
    """
    # do not add the following to the computation graph
    # pdb.set_trace()
    # residual motion on static scene


    x = x.detach()
    # d=abs(x).max()
    d=0.1
    # print("maximum d", d)
    
    x[x>=0] = x[x>=0]/d
    x[x<0] = x[x<0]/d

    return x

def save(path, res_trans):
    # pdb.set_trace()
    res_trans = res_trans[0,0]
    d=abs(res_trans).max()
    res_trans = normalize_trans(res_trans)
    res_trans = res_trans.squeeze().detach().cpu().numpy()
    fig = plt.figure(0)
    plt.imshow(res_trans, vmax=-0.1, vmin=0.1, cmap='bwr')
    plt.colorbar()
    plt.savefig(path)
    plt.close(fig)

class DepthInference:
    def __init__(self, args):
        self.args = args
        self._init_evaluator()
        self._init_img_processor()
        self._init_data_reader()

    def _init_evaluator(self):
        model_opt = get_model_opt(self.args.model_path)
        self.evaluator = EVALUATORS[model_opt.method](model_opt)

    def _init_img_processor(self):
        model_h, model_w = self.evaluator.get_training_res()
        self.img_processor = ImageProcessor(
                self.args.trim, self.args.crop, model_h, model_w)

    def _init_data_reader(self):
        """Make an image iterator
        """
        fp = self.args.input_path

        if fp.endswith('mp4'):
            raise NotImplementedError('doesnot support .mp4 files currently')
        else:
            self.data_reader = SeqReader(fp)

    def infer(self):
        frame_results = {}
        fps = self.data_reader.get_fps()

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        for idx, imgs in enumerate(self.data_reader):
            tgt_img, tgt_img_with_raw_ar = self.img_processor.process(imgs[0])
            src_img, src_img_with_raw_ar = self.img_processor.process(imgs[1])
            disp_colormap, depth = self.evaluator.estimate_depth(tgt_img)
            # pdb.set_trace()
            t2s_trans, t2s_res_trans = self.evaluator.estimate_motion(tgt_img, src_img)
            s2t_trans, s2t_res_trans = self.evaluator.estimate_motion(src_img, tgt_img)
            print('----------------------------------------------------\n')
            print('t=>s ego trans\n', t2s_trans)
            print('t=>s res trans, max %.5f, min %.5f' 
                % (t2s_res_trans[:,0,:,:].max(), t2s_res_trans[:,0,:,:].min()))
            print('s=>t ego trans\n', s2t_trans)
            print('s=>t res trans, max %.5f, min %.5f' 
                % (s2t_res_trans[:,0,:,:].max(), s2t_res_trans[:,0,:,:].min()))
            
            disp_img = concat_depth_img(disp_colormap, tgt_img_with_raw_ar,
                                        self.args.crop[2])
            disp_img_bgr = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
            # cv2.imshow('disp_img', disp_img_bgr)
            if fps is not None:
                cv2.waitKey(int(1000/fps))
            else:
                # use fps 10 if not available
                fps = 10
                cv2.waitKey(int(1000/fps))
            cv2.imwrite(
                    os.path.join(self.args.output_dir, f'{idx:010d}.png'),
                    disp_img_bgr
                    )
            save(
                os.path.join(self.args.output_dir,  f'{idx:010d}_resmof.png'),
                t2s_res_trans
                )

if __name__ == '__main__':
    depth_estimator = DepthInference(InferOptions().parse()[0])
    depth_estimator.infer()
