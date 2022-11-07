import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from copy import copy
from .make_domain import MakeDomain

class GeneticAlgorithm(MakeDomain):
    
    # ------------------------- GA ------------------------------
    def _choose_data_pd(self, data, columns):
        """
        data (pandas): Input data
        columns (list): Column number or column name of the specified column
        """
        if all([type(x) == int for x in columns]):
            choose_data = data.iloc[:, columns]
        elif all([type(x) == str for x in columns]):
            choose_data = data.loc[:, columns]
        else:
            raise ValueError("Use same type or correct type")

        return choose_data
    
    def _mutDomain(self, individual, domain_list, indpb=0.05):
        """GA mutation function. Mutate according to the domain definition.
        Any mutation can be set.
        domain_list (list) : domain.values to list, choice2 without index.
        """
        
        for i in range(len(domain_list)):
            if random.random() < indpb:
                # individual is i+1 because first chromosome of invividual is (0,0), 
                # for dealing with bugs in deap that cannot handle tuples unless the tuple is first.
                if domain_list[i][0]=='randint':
                    individual[i+1] = eval('random.randint{}'.format(domain_list[i][1]))
                elif domain_list[i][0]=='uniform':
                    individual[i+1] = eval('random.uniform{}'.format(domain_list[i][1]))        
                elif domain_list[i][0]=='choice':
                    individual[i+1] = eval('random.choice({})'.format(domain_list[i][1]))        

                elif domain_list[i][0]=='choice2':
                    # When a choice2 function is included
                    #if isinstance(domain_list[i][1], tuple) or isinstance(domain_list[i][1],list):
                        
                    individual[i+1] = eval('random.choice({})'.format(domain_list[i][1]))  
                    # When choice2 index is included
                    #else:
                    #    pass

                elif domain_list[i][0]=='randrange':
                    individual[i+1] = eval('random.randrange{}'.format(domain_list[i][1])) 

        return individual,

    def _cxTwoPointCopy(self, ind1, ind2):
        """GA two-point intersection function. Created because numpy.ndarray is not standard"""
        size = len(ind1)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
            
        return ind1, ind2
    
    
    def _individual_flatten(self,individual):
        """Function to flatten tuples in choice2 when evaluating individual"""
        ind = []
        ind_append = ind.append
        for inner in individual:
            if isinstance(inner, tuple) or isinstance(inner, list):
                for flatten in inner:
                    ind_append(flatten)
            else:
                ind_append(inner)
        return np.array([ind])
    
    def _individual_predict_ml(self, individual, model_list):
        """
        Function that individual(exp) → mid or obj.
        individual : (deap individual)
        model_list : model_list made by MakeMLModel
        """
        # Expand from individual to one line of df
        individual_ = np.delete(individual, 0)  # Drop the very first dummy
        individual_ = self._individual_flatten(individual_)  # flatten the choice2 tuple
        if model_list[0]==None or model_list[0]==[] or model_list[0]==False:  # When mid is None

            obj_eval = np.empty((1,0))
            for i in range(len(model_list[1])):
                obj_eval = np.append(obj_eval, model_list[1][i].predict(individual_) )
            original = list(obj_eval.flatten())  # Calculated value of obj
        else:
            mid = np.empty((1,0))
            for i in range(len(model_list[0])):
                mid = np.append(mid, model_list[0][i].predict(individual_) )
            obj_eval = np.empty((1,0))
            for j in range(len(model_list[1])):
                obj_eval = np.append(obj_eval, model_list[1][j].predict(mid.reshape(1,-1)))
            original = list(mid.flatten()) + list(obj_eval.flatten())  # Calculated mid and obj values
        return obj_eval, individual, original
    
    def _evaluate_equality(self, individual, model_list, weights, obj_scaler,is_standard_scale, ga_function):
        """Function that evaluates an individual in the case of a multipurpose function (default). Returns the objective variable as an evaluation function.
        model_list :(list of learning models) The shape is [[exp→mid],[mid→obj]].
        When mid is None, None is entered in [exp → mid].
        
        weights : (dummy) To match arguments with _evaluate_specific.
        obj_scaler : (dummy) To match arguments with _evaluate_specific.
        is_standard_scale : (dummy) To match arguments with _evaluate_function.
        ga_function : (dummy) To match arguments with _evaluate_function.
        """
        obj_eval, individual, original = self._individual_predict_ml(individual, model_list)
        individual.original = original  # Save to individual original instance

        return tuple(obj_eval.flatten())
    
    def _evaluate_specific(self, individual, model_list, weights, obj_scaler, is_standard_scale, ga_function):
        """A function that evaluates individuals when their weights are complex (default). The target variable is the StandardScaler and the weight is added to the total.
        model_list :(list of learning models) The shape is [[exp→mid],[mid→obj]].
        When mid is None, None is entered in [exp → mid].
        
        weights : (dummy) To match arguments with _evaluate_specific.
        obj_scaler : (fitting StandardScaler) Standardize to evaluate the objective variable of obj equally.
        is_standard_scale : (bool) True : standard scale
        ga_function : (dummy) To match arguments with _evaluate_function.
        """      
        obj_eval, individual, original = self._individual_predict_ml(individual, model_list)
        individual.original = original  # Save to individual original instance
        
        if is_standard_scale:
            obj_eval = obj_scaler.transform([obj_eval])
        weights_eval = obj_eval*weights
        specific_eval = sum(weights_eval.flatten())    
            
        return  specific_eval,

    def _evaluate_function(self, individual, model_list, weights, obj_scaler, is_standard_scale, ga_function):
        '''A function that evaluates an individual with a function. The target variable is the StandardScaler and the weight is added to the total.
        model_list :(list of learning models) The shape is [[exp→mid],[mid→obj]].
        When mid is None, None is entered in [exp → mid].
        weights : (dummy) To match arguments with _evaluate_specific.
        obj_scaler : (fitting StandardScaler) Standardize to evaluate the objective variable of obj equally.
        is_standard_scale : (bool) True : standard scale
        ga_function : (function) Function to apply. 
        '''
        obj_eval, individual, original = self._individual_predict_ml(individual, model_list)
        individual.original = original  # Save to individual original instance
        
        if is_standard_scale:
            obj_eval = obj_scaler.transform([obj_eval])
            obj_eval = obj_eval.reshape(-1,)
        specific_eval =  ga_function(*obj_eval)

        return  specific_eval,
    
    def inverse_fit(self, data, domain='default',
                    NPOP=100, NGEN=50, CXPB=0.5, MUTPB=0.5, weights='equality', evaluate_function='default', is_standard_scale=True,
                    model_list='self', exp='self', obj='self', mid='self'):
        """
        data : (pandas or numpy) Dataframe of making model. To use StandardScaler. Almost no use.
        domain : (dictionary) The values that exp can take in GA are set in a dictionary. The default is uniform for the minimum and maximum values of exp.
        For the detailed explanation
        evaluate_function : (function) A function that calculates the optimization. Takes as many arguments as there are objective variables and has one return value.
        If 'default', using weights that all weights are equal.
        weight : (tuple) Optimization function weights.Takes as many arguments as there are objective variables. If default, equal weight maximization (1.0,1.0, ...) is imposed.
        exp : (list) Specify the experimental condition column as a list of numbers (later column names are also supported).
        obj : (list) Specify the target variable column as a list of numbers (later column names are also supported).
        mid : (list) Specify columns of intermediate generation variables as a list of numbers (the column names are also supported later). 
        If 'self', use information stored in the class instance.
        NPOP : number of population
        NGEN : number of genelation
        CXPB : Crossing probability
        MUTPB : Mutation probability

        """
        # When exp, obj, mid, etc. are not specified, the one used when the model is fit is used
        if exp=='self' and hasattr(self, 'exp'):
            exp = self.exp
        elif exp=='self' and not hasattr(self, 'exp'):
            raise ReferenceError('exp is not defined in class')
        if obj=='self' and hasattr(self, 'obj'):
            obj = self.obj
        elif obj=='self' and not hasattr(self, 'obj'):
            raise ReferenceError('obj is not defined in class')
        if mid=='self' and hasattr(self, 'mid'):
            mid = self.mid
        elif mid=='self' and not hasattr(self, 'mid'):
            raise ReferenceError('mid is not defined in class')
        
        # Read data and df of exp, obj, mid columns
        data = pd.DataFrame(data)
        #exp_df = self._choose_data_pd(data, exp)  # not using
        obj_df = self._choose_data_pd(data, obj)  # using StandardScaler
        
        if mid is not None:
            mid_df = self._choose_data_pd(data, mid)  # not using
        
        # Create default domain
        if domain=='default':
            domain = self.make_domain_auto(data, exp)
            
        # Sort domains in key order (this guarantees the order of domain for loops)        
        domain_list = self.domain_dict_to_list(domain)
        self.domain_list = domain_list

        
        # load model list in class instance made in make_model
        if model_list=='self' and hasattr(self, 'model_list'):
            # If it is defined in self, it is assigned as it is
            model_list = self.model_list
        elif model_list=='self' and not hasattr(self, 'model_list'):
            # If it is not defined, enter the default value for now.
            raise ReferenceError('model_list is not defined in class')
            
        # StandardScaler the obj column. Used in GA evaluation function of using weights.
        if is_standard_scale:
            obj_scaler = StandardScaler()
            obj_scaler.fit(obj_df)
        else:
            obj_scaler = None

        # Defaults are all created with an equal weight of 1.0
        if not (isinstance(weights, tuple) or isinstance(weights, list)):
            weights = tuple([1.0 for i in range(len(obj))])
        
        # Save the original weights.
        self.weights = weights

        # Multi-objective optimization is Not accept except 1.0 or -1.0 or 0. 
        # If If a value other than 1.0 is entered in weights,
        # we will consider the weight by making a single-objective optimization multiplied by that ratio.
        if evaluate_function=='default':
            if all([abs(x) ==1.0 or x ==0.0 for x in weights]):
                evaluate_function = self._evaluate_equality
            else:
                evaluate_function = self._evaluate_specific
            
            # weights setting
            if all([abs(x) ==1.0 or x ==0.0 for x in weights]):
                specific_weight = None
            else:
                specific_weight = copy(weights)
                weights = (1.0, )
            
            ga_function=None
        
        else:  # When a function is entered.
            ga_function = evaluate_function
            evaluate_function = self._evaluate_function
            
            specific_weight = copy(weights)
            weights = (1.0, ) 
        
        #ga_function = np.frompyfunc(ga_function, len(obj), 1)  # ind is 1 row, don't need
        
        # save info for post process
        self.ga_function = ga_function
        self.is_standard_scale = is_standard_scale
        
        # ---------------------------- Create a function to make GA chromosome---------------------------------
        
        creator.create('FitnessMulti', base.Fitness, weights=weights )
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        
        # Create a function to create chromosomes along the domain dictionary.
        # choice2_key_save=-1
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

                
        # ---------------------------- Creating functions to create GA individuals and populations ------------------------------
        
        # For some reason with random.choice, if you do not put the tuple first, 
        # you will get an error when you later insert rancom.choice.
        # I wrote in the issue of Github of deap.
        # defining attributes for individual
        toolbox.register("dummy", 
                         random.choice, ((0,0),(0,0)) )
        
        individual_function=[toolbox.dummy]
        for i in domain.keys():
            # Added only when it is not a choice2 body
            if not ((domain[i][0]=='choice2') and (isinstance(domain[i][1], int) or isinstance(domain[i][1],float)) ):
                individual_function.append(eval('toolbox.exp{}'.format(i)))
        individual_function = tuple(individual_function) 

        
        # register attributes to individual
        toolbox.register('individual', tools.initCycle, creator.Individual,
                         individual_function,
                          n = 1)
        
        # individual to population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        
        # -----------------------Register the function of GA evolution calculation part----------------------------
        #if evaluate_function=='equality':
        #    evaluate_function=self._evaluate_equality
            
        # evolution
        toolbox.register('mate', self._cxTwoPointCopy)
        toolbox.register('mutate', self._mutDomain, domain_list=domain_list, indpb = 0.05)
        toolbox.register('select', tools.selTournament, tournsize=3)
        toolbox.register('evaluate', evaluate_function,
                         model_list=model_list, weights=specific_weight, obj_scaler=obj_scaler,
                        is_standard_scale=is_standard_scale, ga_function=ga_function)
        
        # -----------------------GA main loop-------------------------------------
        def main(NPOP, NGEN, CXPB, MUTPB):
            """A function that turns the main loop of GA. Also, since we want to use the variable in the inverse_fit function, we do not divide the function.
            What I don't do with ga.algorithms is to save all the pops."""
            
            # Set to output a fixed random number
            random.seed(64)
            # Generate initial population
            pop = toolbox.population(n=NPOP)
            # Should be included in the argument            
            pop_save = []
            pop_save_append = pop_save.append
            pop_save_append(copy(pop))

            print("Start of evolution")
        
            # Initial population assessment
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
        
            print("  Evaluated %i individuals" % len(pop))
            
            # Evolutionary calculation started
            for g in range(NGEN):
                print("-- Generation %i --" % g)
        
                # Select next generation population
                offspring = toolbox.select(pop, len(pop))
                # Create a clone of a population
                offspring = list(map(toolbox.clone, offspring))
        
                # Adapt crossover and mutation to selected populations
                # Extract even-numbered and odd-numbered individuals and intersect (even for [:: 2], odd for [1 :: 2])
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)
                        # Removed fitness.values (evaluation values) for replaced individuals
                        del child1.fitness.values
                        del child2.fitness.values
        
                for mutant in offspring:
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
        
                # Collect the individuals whose fitness is not calculated and calculate the fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
        
                print("  Evaluated %i individuals" % len(invalid_ind))
    
                # Make the next generation group offspring
                pop[:] = offspring
                
                # saving pop
                pop_save_append(copy(pop))
                
                # Array the fitness of all individuals
                fits = [ind.fitness.values[0] for ind in pop]
                                
                length = len(pop)
                mean = sum(fits) / length
                sum2 = sum(x*x for x in fits)
                std = abs(sum2 / length - mean**2)**0.5
        
                print("  Min %s" % min(fits))
                print("  Max %s" % max(fits))
                print("  Avg %s" % mean)
                print("  Std %s" % std)
        
            print("-- End of (successful) evolution --")
        
            best_ind = tools.selBest(pop, 1)[0]
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            return best_ind,best_ind.fitness.values, pop, pop_save

        best_ind, best_ind_value, pop, pop_save  = main(NPOP, NGEN, CXPB, MUTPB)
        
        return best_ind, best_ind_value, pop, pop_save
    
    
    # -----------------------------The end of the GA------------------------------------
    
    
    def pop_data_fitness_value(self, pop_save, weights='self', ga_function='self'):
        """
        Function to restore original df (exp, mid, obj) from pop_save.
        Use original information and fitness information stored as ind instances
        """
        # Only need weights for multi-purpose
        if weights=='self' and hasattr(self, 'weights'):
            weights = self.weights
        elif weights=='self' and not hasattr(self, 'weights'):
            weights = tuple([1.0 for i in range(len(obj))])

        if ga_function=='self' and hasattr(self, 'ga_function'):
            ga_function = self.ga_function
        elif ga_function=='self' and not hasattr(self, 'ga_function'):
            ga_function = None

        if ga_function==None:
            if all([abs(x) ==1.0 or x ==0.0 for x in weights]):
                mode='equality'
            else:
                mode='specific'  
        else:
            mode='function'

        len_fitness_value = len(pop_save[0][0].fitness.values)

        shape_list = []
        shape_list_append = shape_list.append
        for pop in pop_save:
            for ind in pop:
                ind_ = np.delete(ind, 0)  # Drop the dummy (0,0).
                exp = self._individual_flatten(ind_).tolist()[0]  # List of experimental conditions
                midobj = ind.original


                if mode=='equality':
                    values = np.array(ind.fitness.values)
                    values_ = values*weights # Multiply each value by weights.
                    values_[np.isnan(values_)] = 0
                    fitness = [values_.sum().astype(float)]

                elif mode=='specific':  # For weighted single-objective optimization

                    fitness = list(ind.fitness.values)

                elif mode=='function':

                    fitness = list(ind.fitness.values)
                data_type = ['predict']
                shape = exp + midobj + data_type +fitness
                shape_list_append(shape)

        return pd.DataFrame(shape_list)


    def raw_data_fitness_value(self, data, obj='self', weights='self', ga_function='self', is_standard_scale='self'):
        """Add fitness values for raw data, By this function, you can know raw data fitness values same as pop from GA. 
        Args:
            data (pandas.DataFrame): used raw data.
            obj : (list of int) Specify the target variable column as a list of numbers (later column names are also supported).
            weights (str, optional): using ga evaluate weights. Defaults to 'self'.
            ga_function (str, optional): using ga_evalulate function. Defaults to 'self'.
            is_standard_scale (str, optional): using when ga_evaluate standard scaler. Defaults to 'self'.
        
        Returns:
            (pandas.DataFrame)
            add fitness values data
        """
         # Only need weights for multi-purpose
        if weights=='self' and hasattr(self, 'weights'):
            weights = self.weights
        elif weights=='self' and not hasattr(self, 'weights'):
            weights = tuple([1.0 for i in range(len(obj))])

        if ga_function=='self' and hasattr(self, 'ga_function'):
            ga_function = self.ga_function
        elif ga_function=='self' and not hasattr(self, 'ga_function'):
            ga_function = None

        if is_standard_scale=='self' and hasattr(self, 'is_standard_scale'):
            is_standard_scale = self.is_standard_scale
        elif is_standard_scale=='self' and not hasattr(self, 'is_standard_scale'):
            is_standard_scale = False
            
        if obj=='self' and hasattr(self, 'obj'):
            obj = self.obj

        if ga_function==None:
            if all([abs(x) ==1.0 or x ==0.0 for x in weights]):
                mode='equality'
            else:
                mode='specific'  
        else:
            mode='function'


        data_obj = data.iloc[:,obj]
        obj_scaler = StandardScaler()

        if mode=='equality':

            values = np.array(data_obj)
            values_ = values*weights # Multiply each value by weights.
            values_[np.isnan(values_)] = 0
            fitness = values_.sum(axis=1).astype(float)

        elif mode=='specific':  # For weighted single-objective optimization
            if is_standard_scale:
                obj_scaler = StandardScaler()
                values = obj_scaler.fit_transform(data_obj)
            else:
                values = np.array(data_obj)
            values_ = values*weights # Multiply each value by weights.
            values_[np.isnan(values_)] = 0
            fitness = values_.sum(axis=1).astype(float)

        elif mode=='function':
            ga_function = np.frompyfunc(ga_function, len(obj), 1)
            if is_standard_scale:
                obj_scaler = StandardScaler()
                values = obj_scaler.fit_transform(data_obj)
            else:
                values = np.array(data_obj)
            fitness = ga_function(values)

        len_data_obj = data_obj.shape[0]
        data_type = pd.DataFrame([['raw'] for x in range(len_data_obj)],columns=['data_type'])
        fitness = pd.DataFrame(fitness, columns=['fitness_value'])

        data_fitness = pd.concat([data.reset_index(drop=True), data_type, fitness], axis=1)
        
        return data_fitness
    
    def ga_restore(self, pop_save, data, obj='self', dummies_list='self',categories_reborn='self',
                        is_raw_data = True, is_sort=True, is_duplicate_drop=True, is_origin_shape=True):
        """Function to completely restore pop_save.
        pop_save : (list) Saved pop by inverse_fit
        data_columns : (list) Data column name after dummy variable conversion.
        
        It doesn't work without the information of self.dummies and self.categories converted by CategoricalPreprocessing's formatting.
        Also, use CategoricalPreprocessing inverse_formatting.
        """

        if dummies_list=='self' and hasattr(self, 'dummies_list'):
            # If it is defined in self, it is assigned as it is
            dummies_list = self.dummies_list
        elif dummies_list=='self' and not hasattr(self, 'dummies_list'):
            # If it is not defined, enter the default value for now.
            dummies_list = []
        if categories_reborn=='self' and hasattr(self, 'categories_reborn'):
            categories_reborn = self.categories_reborn
        elif categories_reborn=='self' and not hasattr(self, 'categories_reborn'):
            categories_reborn = [None, None]
            
        if obj=='self' and hasattr(self, 'obj'):
            obj = self.obj

        # later self.dummies and self.categories fixes.
        data_columns = data.columns
        columns = list(data_columns) + ['data_type'] + ['fitness_value']

        pop_fit_data = self.pop_data_fitness_value(pop_save, weights='self', ga_function='self')
        pop_fit_data.columns = columns

        if is_raw_data:
            raw_fit_data = self.raw_data_fitness_value(data, obj, weights='self', ga_function='self', is_standard_scale='self')
            raw_fit_data.columns = columns
            fit_data = pd.concat([pop_fit_data, raw_fit_data], axis=0)
        else:
            fit_data = pop_fit_data

        # Restore.
        if is_origin_shape:
            fit_data = self.inverse_formatting(fit_data, dummies_list=dummies_list, categories_reborn=categories_reborn)
            
        if is_sort:
            fit_data = fit_data.sort_values(by=["fitness_value"], ascending=False)
        if is_duplicate_drop:
            fit_data = fit_data.drop_duplicates()
        
        return fit_data