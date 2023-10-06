from typing import Optional
from contextlib import contextmanager
from copy import deepcopy
import torch

@contextmanager
def _perturbed_model(
  model,
  sigma: float,
  rng,
  magnitude_eps: Optional[float] = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model

# https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
@torch.no_grad()
def _pacbayes_sigma(
  model,
  dataloader,
  accuracy: float,
  seed: int,
  magnitude_eps: Optional[float] = None,
  search_depth: int = 15,
  montecarlo_samples: int = 10,
  accuracy_displacement: float = 0.1,
  displacement_tolerance: float = 1e-2,
) -> float:
  lower, upper = 0, 2
  sigma = 1

  BIG_NUMBER = 10348628753
  device = next(model.parameters()).device
  rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
  rng.manual_seed(BIG_NUMBER + seed)

  displacement_list, sigma_list = [],[] # remove
  for _ in range(search_depth):
    sigma = (lower + upper) / 2
    accuracy_samples = []
    for _ in range(montecarlo_samples):
      with _perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
        loss_estimate = 0
        for data, target in dataloader:
          # data = data.to(device, dtype=torch.float)
          # modified line to the line below to move target to device
          data, target = data.to(device, dtype=torch.float), target.to(device)
          logits = p_model(data)
          pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
          batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
          loss_estimate += batch_correct.sum()
        loss_estimate /= len(dataloader.dataset)
        accuracy_samples.append(loss_estimate)
    displacement = abs(np.mean(accuracy_samples) - accuracy)
    displacement_list.append(displacement) # remove
    sigma_list.append(sigma) # remove
    if abs(displacement - accuracy_displacement) < displacement_tolerance:
      break
    elif displacement > accuracy_displacement:
      # Too much perturbation
      upper = sigma
    else:
      # Not perturbed enough to reach target displacement
      lower = sigma
      
  #pol = np.polyfit(np.log(sigma_list),np.log(displacement_list),1) # remove
  #return sigma,pol[0]
  return sigma_list,displacement_list

#%%

@contextmanager
def _perturbed_model_deterministic(
  model,
  r: float
):
  model = deepcopy(model)
  try:
    [p.add_(r) for p in model.parameters()]
    yield model
  finally:
    [p.sub_(r) for p in model.parameters()]
    del model

@torch.no_grad()    
def variogram(model, dataloader, accuracy):
    
    device = next(model.parameters()).device
    
    #delta_list, r_list = [],[1/64,1/128,1/256,1/512,1/1024,1/2048]
    delta_list, r_list = [],[1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256,1/512,1/1024,1/2048]
    for i in range(len(r_list)):
        r = r_list[i]
        accuracy_samples = []
        with _perturbed_model_deterministic(model, r) as p_model:
            loss_estimate = 0
            for data, target in dataloader:
                # data = data.to(device, dtype=torch.float)
                # modified line to the line below to move target to device
                data, target = data.to(device, dtype=torch.float), target.to(device)
                logits = p_model(data)
                pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
                batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
                loss_estimate += batch_correct.sum()
            loss_estimate /= len(dataloader.dataset)
            accuracy_samples.append(loss_estimate)
        displacement = abs(np.mean(accuracy_samples) - accuracy)
        delta_list.append(displacement) # remove
        
    #pol = np.polyfit(np.log(r_list),np.log(delta_list),1)
    #return pol[0]
    return delta_list

@torch.no_grad()    
def variogram2(model, dataloader, accuracy):
    
    device = next(model.parameters()).device
    
    delta_list, r_list = [],[1/64,1/128,1/256,1/512,1/1024,1/2048]
    for i in range(len(r_list)):
        r = r_list[i]
        displacement = 0
        with _perturbed_model_deterministic(model, r) as p_model:
            for data, target in dataloader:
                data, target = data.to(device, dtype=torch.float), target.to(device)
                logits1 = model(data)
                logits2 = p_model(data)
                displacement += torch.sum(torch.square(torch.sub(logits1,logits2)))
        displacement /= len(dataloader.dataset)
        delta_list.append(displacement.cpu().numpy())
        
    pol = np.polyfit(np.log(r_list),np.log(delta_list),1)
    return pol[0]

#%%

def box_counting(model):
    
    blacklist = {'bias', 'bn'}
    weights_temp = [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]
    weights = torch.cat([p.view(-1) for p in weights_temp], dim=0).detach().cpu().numpy()
    
    #weights = weights[np.abs(weights)<3*np.std(weights)] # removing outliers
    
    weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights)) # normalization

    r_list = 2**(np.arange(10))
    nbr_boxes = []    
    for delta in r_list:
        weights_reduced = weights//(1/delta)
        nbr_boxes.append(len(np.unique(weights_reduced)))

    pol = np.polyfit(np.log(r_list),np.log(np.array(nbr_boxes)),1)
    
    return pol[0]  

#%%

import numpy as np
from pandas import DataFrame
import torch

import glob
#path = '.\\saved_models_svhn\\'
path = '.\\saved_models_cifar\\'
parameters_string = []
files_test_error = glob.glob(path + 'test_error*')
for i in range(len(files_test_error)):
    temp_string = files_test_error[i].partition('test_error')
    parameters_string.append(temp_string[2])

train_error,test_error = [],[]
measure_list,measure_list2 = [],[]
for param_str in parameters_string:
    train_error_temp = np.load(path+'train_error'+param_str)
    train_error.append(train_error_temp)
    test_error.append(np.load(path+'test_error'+param_str))
    temp_model = torch.load(path+'model'+param_str[:-3]+'pth')
    temp_trainloader = torch.load(path+'trainloader'+param_str[:-3]+'pth')
    #sigma,FD = _pacbayes_sigma(temp_model, temp_trainloader, 1-train_error_temp, 0, accuracy_displacement=0.05)
    FD = variogram(temp_model, temp_trainloader, 1-train_error_temp)
    #FD = variogram2(temp_model, temp_trainloader, 1-train_error_temp)
    #FD = box_counting(temp_model)
    """for thresh in [0.025,0.05,0.075,0.1,0.25,0.5]:
        sigma_list,displacement_list = _pacbayes_sigma(temp_model, temp_trainloader, 1-train_error_temp, 0, accuracy_displacement=thresh)
        measure_list.append(sigma_list)
        measure_list.append(displacement_list)"""
    #measure_list.append(1 / sigma ** 2)
    measure_list2.append(FD) # remove
    
train_error=np.array(train_error)
test_error=np.array(test_error)
    
"""all_measures = DataFrame(measure_list2)

import scipy.stats as stats

# Kendall's rank-correlation coefficients
print('Correlations Between Compexity Measures and Generalization Gap (Test Error-Train Error):\n')
tau_gen_gap, p_value = stats.kendalltau(test_error-train_error, all_measures)
print(tau_gen_gap)"""

import scipy.stats as stats

measure_list = read_list('sigma_displacement_list_cifar')
#measure_list = read_list('sigma_displacement_list_svhn')
#measure_list = read_list('sigma_displacement_list_mnist')
#npz_file = np.load('train_test_error_svhn.npz') # SVHN
#npz_file = np.load('train_test_error_mnist.npz') # MNIST
#train_error,test_error = npz_file['arr_0'],npz_file['arr_1'] #SVHN
tau_gen_gap_list = []
for i in range(0,12,2):
    FD_list = []
    for j in range(0,348,12): # CIFAR
    #for j in range(0,252,12): # SVHN
    #for j in range(0,216,12): # MNIST
        pol = np.polyfit(np.log(measure_list[i+j][2:]),np.log(np.array(measure_list[i+j+1][2:])**2),1)
        FD_list.append(1/pol[0])
    all_measures = DataFrame(FD_list)
    tau_gen_gap, p_value = stats.kendalltau(test_error-train_error, all_measures)
    tau_gen_gap_list.append(tau_gen_gap)
    
pac_bayes_list = []
#for i in range(6,348,12): # CIFAR
for i in range(6,252,12): # SVHN
    pac_bayes_list.append(1/(measure_list[i][-1]**2))
all_measures = DataFrame(pac_bayes_list)
tau_gen_gap, p_value = stats.kendalltau(test_error-train_error, all_measures)
print(tau_gen_gap)

r_list = [1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256,1/512,1/1024,1/2048]
FD_list = []
#for i in range(29): # CIFAR
for i in range(21): # SVHN
    pol = np.polyfit(np.log(r_list),np.log(np.array(measure_list2[i])),1)
    FD_list.append(1/pol[0])
all_measures = DataFrame(FD_list)
tau_gen_gap, p_value = stats.kendalltau(test_error-train_error, all_measures)