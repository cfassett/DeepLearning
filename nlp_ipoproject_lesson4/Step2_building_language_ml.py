from fastai.text import *
# I did this outside the jupyter notebook because it was before I figured out how to run a jupyter notebook on the GPU node I used for this assignment.  
# Step2_plot.png shows the typical learning rate graph we've come to expect.
# sorry this does not record the language model accuracy, but my recollection was it was ~40%.

path='/users/cfassett/data/ml_experiments/build/'
#data_lm = TextLMDataBunch.from_folder(path)
#data_lm.save()
data_lm = load_data(path)
bs=16
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
#learn.lr_find()
#fig=learn.recorder.plot(return_fig=True)
#fig.savefig('plot.png')
#learn.fit_one_cycle(1,3e-2,moms=(0.8,0.7))
#learn.save('fit_head');
#learn.load('fit_head');
#learn.unfreeze()
learn.load('fine_tuned_v3')
learn.unfreeze()
learn.fit_one_cycle(2,3e-3, moms=(0.8,0.7))
learn.save('fine_tuned_v4')
