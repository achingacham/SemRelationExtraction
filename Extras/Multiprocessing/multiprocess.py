

'''

import multiprocessing
import os

def worker(d, key, value):
    
    if key%2:
        
        if "odd" in d:
            d["odd"] += [key]
        else:
            d["odd"] = [key]
            
    if key%2==0:
        
        if "even" in d:
            d["even"] += [key]
        else:
            d["even"] = [key]
    
    print(">>>>>>>>>>",os.getpid(), key, d)
    
if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    d = mgr.dict()
    with multiprocessing.Pool(2) as pool:
        
        
        jobs = [pool.apply_async(worker, args=(d, i, i*2,)) for i in range(10)]
        
        pool.close()
        pool.join()
        
    print('Results:', d)
    
'''

'''
from multiprocessing import Pool
from multiprocessing import Process, Manager
import os



def f(dict_process, x):
    
    k = 1
    
    while(k<=x):
        
        
        if x%k == 0:
            
            
            
            if x in dict_process:
                t = dict_process[x]
                t += [str(k)]
                dict_process[x] = t

            else:
                dict_process[x] = [str(k)]

        k += 1   
    
if __name__ == '__main__':
    
    #with Pool(5) as p:
    #    individual_dict = p.map(f, [1, 2, 3]) 
    #    print(individual_dict)
    
    #for i in range(10)[1:]:
    #    p = Process(target=f1, args=(i,))    
    #    p.start()
        
    #p.join()
    
    
    with Manager() as manager:
        
        dict_process = manager.dict()
        
        cpu = os.cpu_count()
        #lock = manager.Lock()
        
        with Pool(cpu) as pl:
            
            for i in range(100)[1:]:
                
                p = pl.apply_async(f, args=(dict_process, i,))
                
                #p.start()
                #p.join()
            
            pl.close()
            pl.join()
            
        print(dict_process)
        
'''