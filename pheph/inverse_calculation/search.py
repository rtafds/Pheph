import time
from deap import base, creator

class Search(MakeDomain):

    def _make_random_function(self, domain):
        toolbox = base.Toolbox()
        
        creator.create('FitnessDummy', base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessDummy)
        
        domain = dict(sorted(domain.items(), key=lambda x:x[0]))

        for i in domain.keys():
            if type(domain[i]) in (str, int, float):
                # When something strange comes in like {1:3}
                raise ValueError("Please Enter correct value at domain in {}".format(i))

            else:
                if domain[i][0]=='randint':
                    toolbox.register('exp{}'.format(i), 
                            random.randint, int(domain[i][1][0]), int(domain[i][1][1]))

                elif domain[i][0]=='uniform':
                    toolbox.register('exp{}'.format(i), 
                            random.uniform, domain[i][1][0], domain[i][1][1])

                elif domain[i][0]=='choice':
                    toolbox.register('exp{}'.format(i), 
                            random.choice, domain[i][1])   

                elif domain[i][0]=='choice2':
                    # If there is a choice2 body.
                    if isinstance(domain[i][1], tuple) or isinstance(domain[i][1],list):
                        toolbox.register('exp{}'.format(i), 
                            random.choice, domain[i][1])
                    # If there is a choice2 index only(dummy)
                    elif isinstance(domain[i][1], int) or isinstance(domain[i][1],float):
                        pass
                    else:
                        raise ValueError("Please Enter correct value at domain in {}".format(i))


                elif domain[i][0]=='randrange':
                    toolbox.register('exp{}'.format(i), 
                            random.randrange, domain[i][1][0],domain[i][1][1],domain[i][1][2])
                else:
                    raise ValueError("domain have an unexpected value")

        random_func = []
        for i in domain.keys():
            # Added only when it is not a choice2 body
            if not ((domain[i][0]=='choice2') and (isinstance(domain[i][1], int) or isinstance(domain[i][1],float)) ):
                random_func.append(eval('toolbox.exp{}'.format(i)))
        random_func = tuple(random_func) 


        # register attributes to individual
        toolbox.register('random_search', tools.initCycle, creator.Individual,
                         random_func,
                          n = 1)
        return toolbox.random_search

    def _tuple_flatten(self, row):
        """Function to flatten tuples in choice2 when evaluating individual"""
        ind = []
        ind_append = ind.append
        for inner in row:
            if isinstance(inner, tuple) or isinstance(inner, list):
                for flatten in inner:
                    ind_append(flatten)
            else:
                ind_append(inner)
        return np.array([ind])

    def _predict_ml(self, predict_array, model_list):
        # Expand from individual to one line of df
        if model_list[0]==None or model_list[0]==[] or model_list[0]==False:  # When mid is None
            for i in range(len(model_list[1])):
                obj_i = model_list[1][i].predict(predict_array)
                obj_i = obj_i.reshape(-1,1)
                if i==0:
                    obj = obj_i
                else:
                    obj = np.concatenate([obj, obj_i], axis=1)
            returns = np.concatenate([predict_array, obj],axis=1)
        else:
            for i in range(len(model_list[0])):
                mid_i = model_list[0][i].predict(predict_array)
                mid_i = mid_i.reshape(-1,1)
                if i==0:
                    mid = mid_i
                else:
                    mid = np.concatenate([mid, mid_i], axis=1)
            for j in range(len(model_list[1])):
                obj_j = model_list[1][j].predict(mid)
                obj_j = mid_i.reshape(-1,1)
                if j==0:
                    obj = obj_j
                else:
                    obj = np.concatenate([obj, obj_j], axis=1)
            returns = np.concatenate([predict_array, mid, obj],axis=1)
        return returns
    
    def random_search(self, data, domain, model_list='self', n_row=1000, max_time=False, is_origin_shape=True):
        
        # load model list in class instance made in make_model
        if model_list=='self' and hasattr(self, 'model_list'):
            # If it is defined in self, it is assigned as it is
            model_list = self.model_list
        elif model_list=='self' and not hasattr(self, 'model_list'):
            # If it is not defined, enter the default value for now.
            raise ReferenceError('model_list is not defined in class')
        
        first_time = time.time()

        random_function = self._make_random_function(domain)
        data_column = data.columns

        predict_array = self._tuple_flatten(random_function())
        if n_row==False:
            n_row=10000**10
        for i in range(n_row-1):
            if not (max_time==None or max_time<=0 or max_time==False):
                elapsed_time = time.time() - first_time
                if elapsed_time>=max_time:
                    break
            predict_row = self._tuple_flatten(random_function())
            predict_array = np.concatenate([predict_array, predict_row], axis=0)

        predict_search = self._predict_ml(predict_array, model_list)

        predict_search  = pd.DataFrame(predict_search, columns = data_column)
        if is_origin_shape:
            predict_search = self.inverse_formatting(predict_search)
        return predict_search
    
    
    def _make_grid_search_list(self, domain, uniform_split_num=10):
        domain = dict(sorted(domain.items(), key=lambda x:x[0]))

        grid_search_list = []
        grid_search_list_append = grid_search_list.append

        for i in domain.keys():
            if type(domain[i]) in (str, int, float):
                # When something strange comes in like {1:3}
                raise ValueError("Please Enter correct value at domain in {}".format(i))

            else:
                if domain[i][0]=='randint':
                    search_list = [x for x in range(domain[i][1][0], domain[i][1][1] +1 )]
                    search_list_ = list(map(lambda x: [x], search_list))
                    grid_search_list_append(search_list_)

                elif domain[i][0]=='uniform':
                    min_ = domain[i][1][0]
                    max_ = domain[i][1][1]
                    width = max_ - min_
                    step = width/(uniform_split_num-1)
                    search_list = list(np.arange(min_, max_+step, step))
                    search_list_ = list(map(lambda x: [x], search_list))
                    grid_search_list_append(search_list_)

                elif domain[i][0]=='choice':
                    search_list = list(domain[i][1])
                    search_list_ = list(map(lambda x: [x], search_list))
                    grid_search_list_append(search_list_)

                elif domain[i][0]=='choice2':
                    # If there is a choice2 body.
                    if isinstance(domain[i][1], tuple) or isinstance(domain[i][1],list):
                        search_list = list(domain[i][1])
                        grid_search_list_append(search_list)

                    # If there is a choice2 index only(dummy)
                    elif isinstance(domain[i][1], int) or isinstance(domain[i][1],float):
                        pass
                    else:
                        raise ValueError("Please Enter correct value at domain in {}".format(i))

                elif domain[i][0]=='randrange':
                    search_list = [x for x in range(domain[i][1][0], domain[i][1][1], domain[i][1][2])]
                    search_list_ = list(map(lambda x: [x], search_list))
                    grid_search_list_append(search_list_)
                else:
                    raise ValueError("domain have an unexpected value")
        return grid_search_list

    def grid_search(self, data, domain, model_list='self', uniform_split_num=10, max_time=False, is_origin_shape=True):
                
        # load model list in class instance made in make_model
        if model_list=='self' and hasattr(self, 'model_list'):
            # If it is defined in self, it is assigned as it is
            model_list = self.model_list
        elif model_list=='self' and not hasattr(self, 'model_list'):
            # If it is not defined, enter the default value for now.
            raise ReferenceError('model_list is not defined in class')
        
        first_time = time.time()
        data_column = data.columns

        grid_search_list = self._make_grid_search_list(domain_, uniform_split_num=uniform_split_num)
        g = grid_search_list
        len_g = len(g)
        count = [0 for x in range(len_g)]
        max_count=[len(x)-1 for x in g]

        X_grid_search=[]
        X_grid_search_append = X_grid_search.append
        while True:      
            if not (max_time==None or max_time<=0 or max_time==False):
                elapsed_time = time.time() - first_time
                if elapsed_time>=max_time:
                    break

            grid_row=[]
            for i in range(len_g):
                grid_row +=g[i][count[i]]
            X_grid_search_append(grid_row)

            count[0] += 1
            for j in range(len_g): 
                if count[j]>max_count[j]:
                    count[j]=0
                    count[j+1] +=1
            if count[-1]==max_count[-1]:
                break
        X_grid_search = np.array(X_grid_search)

        predict_search = self._predict_ml(X_grid_search, model_list)
        predict_search  = pd.DataFrame(predict_search, columns = data_column)
        if is_origin_shape:  
            predict_search = self.inverse_formatting(predict_search)
        return predict_search