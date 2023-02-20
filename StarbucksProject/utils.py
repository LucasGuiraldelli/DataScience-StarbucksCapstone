import pandas as pd
import missingno as msno

def create_offer_id_col(val):
    '''
    function created with the puporse of interact with all values of a column
    and by that differentiate the first word and then separate into a new column

    input-> val: value of a column
    '''
    if list(val.keys())[0] in ['offer id', 'offer_id']:
        return list(val.values())[0]
    
def create_amount_col(val):
    '''
    function created with the puporse of interact with all values of a column
    and by that differentiate the first word and then separate into a new column
    
    input-> val: value of a column
    '''

    if list(val.keys())[0] in ['amount']:
        return list(val.values())[0]

def first_infos(df: pd.DataFrame):
    '''
    function created with the puporse of get a first look in the dataframe selected
    
    input-> df: dataframe that you want to have a first look
    '''
    print(df.info())
    msno.bar(df, figsize=(5,3), fontsize=15)
    print(df.sample(3))

def create_user_item_matrix(offers_given, filename):
    '''
    Return the user item matrix that indicate the number of offer complete of a particular user
    
    INPUT:
    offer - a cleaned transcript dataframe
    filename(string) - the file name that save the user item matrix
    
    OUTPUT:
    user_item_matrix - the user item matrix which 
        - row is user 
        - column is offer
        - value is the number of offer complete by the user (NaN means no offer given)
    
    '''
    import numpy as np 

    # create an empty user item matrix
    user_item_matrix = offers_given.groupby(['person', 'offer_id'])['event'].agg(lambda x: np.nan).unstack()
    # Retaining only bogo and discount offers first and dropping all else for simplicity
    user_item_matrix.drop(list(portfolio[portfolio['offer_type']=='informational']['id']), axis=1, inplace=True)
    
    for offer_id in user_item_matrix.columns:
        print("Processing Offer: ", offer_id)
        num = 0
        for person in user_item_matrix.index:
            num += 1
            if num % 5000 == 0:
                print("finished upto #:", num, 'persons')
            events = []
            for event in offers_given[(offers_given['offer_id']==offer_id) & (offers_given['person']==person)]['event']:
                events.append(event)
            if len(events) >= 3:
                user_item_matrix.loc[person, offer_id] = 0
                for i in range(len(events)-2):
                    # check that transaction sequence is offer received -> offer viewed -> offer completed
                    # Only if its in that order we assume the offer was successfully accepted
                    if (events[i] == 'offer received') & (events[i+1] == 'offer viewed') & (events[i+2] == 'offer completed'):
                        user_item_matrix.loc[person, offer_id] += 1
            elif len(events) > 0:
                user_item_matrix.loc[person, offer_id] = 0
    
    # store the large martix into file
    fh = open(filename, 'wb')
    pickle.dump(user_item_matrix,fh)
    fh.close()
    
    return user_item_matrix


#Function to split dataset in training and testing subsets
def user_item_train_test_split (transcript_df, training_perc=0.7):
    """Function that prepares train and test user_item_matrices out of transcript dataset.
    
    INPUT: 
    1.  dataframe to split into training and testing datasets
    2.  training dataset size percentage, default = 0.7

    OUTPUT:
    1.  train dataframe
    2.  test dataframe
    3.  train user_item_matrix
    4.  test user_item_matrix
        
    """
    import math
    
     #saving a dataframe copy to work with
    transcript_dfcopy = transcript_df.copy()
    
    training_size= math.ceil(transcript_dfcopy.shape[0]*training_perc)
    testing_size= transcript_dfcopy.shape[0]-training_size
    
    training_df = transcript_dfcopy.head(training_size)
    test_df = transcript_dfcopy.iloc[training_size:training_size+testing_size]
       
    #converting both into user_item_matrix format
    print('Preparing Training matrix')
    train_matrix = create_user_item_matrix(training_df, 'train_df.p')
    print('Preparing Testing matrix')
    test_matrix = create_user_item_matrix(test_df, 'test_df.p')
    
    return training_df,test_df,train_matrix, test_matrix  


def FunkSVD(user_item_matrix, latent_features=4, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization
    
    INPUT:
    user_mat - (numpy array) a matrix with users as rows, offers as columns, and completion as values
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate 
    iters - (int) the number of iterations
    
    OUTPUT:
    user_mat - (numpy array) a user by latent feature matrix
    movie_mat - (numpy array) a latent feature by movie matrix
    '''
    
    # Set up useful values to be used through the rest of the function
    n_users = user_item_matrix.shape[0]
    n_offers = user_item_matrix.shape[1]
    num_offers = np.count_nonzero(~np.isnan(user_item_matrix))
    
    # initialize the user and movie matrices with random values
    user_mat = np.random.rand(n_users, latent_features)
    offer_mat = np.random.rand(latent_features, n_offers)
    
    # initialize sse at 0 for first iteration
    sse_accum = 0
    
    # header for running results
    print("Optimizaiton Statistics")
    print("Iterations | Mean Squared Error ")
    
    # for each iteration
    for iteration in range(iters):

        # update our sse
        old_sse = sse_accum
        sse_accum = 0
        
        # For each user-offer pair
        for i in range(n_users):
            for j in range(n_offers):
                
                # if the offer was completed
                if user_item_matrix[i, j] > 0:
                    
                    # compute the error as the actual minus the dot product of the user and movie latent features
                    diff = user_item_matrix[i, j] - np.dot(user_mat[i, :], offer_mat[:, j])
                    
                    # Keep track of the sum of squared errors for the matrix
                    sse_accum += diff**2
                    
                    # update the values in each matrix in the direction of the gradient
                    for k in range(latent_features):
                        user_mat[i, k] += learning_rate * (2*diff*offer_mat[k, j])
                        offer_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

        # print results for iteration
        print("%d \t\t %f" % (iteration+1, sse_accum / num_offers))
        
    return user_mat, offer_mat 


#Predict reaction function
def predict_reaction(user_matrix, offer_matrix, user_id, offer_id):
    '''
    INPUT:
    user_matrix - user by latent factor matrix
    offer_matrix - latent factor by offer matrix
    user_id - the user_id from the reviews df
    offer_id - the offer_id according the offer df
    
    OUTPUT:
    pred - the predicted reaction for user_id-offer_id according to FunkSVD
    '''
    try:
        # Use the training data to create a series of users and movies that matches the ordering in training data
        user_ids_series = np.array(train_matrix.index)
        offer_ids_series = np.array(train_matrix.columns)

        # User row and offer Column
        user_row = np.where(user_ids_series == user_id)[0][0]
        offer_col = np.where(offer_ids_series == offer_id)[0][0]

        # Take dot product of that row and column in U and V to make prediction
        pred = np.dot(user_matrix[user_row, :], offer_matrix[:, offer_col])

        return pred
    
    except:
        return None


#Generate validation score function
def validation_score(test_matrix, user_mat, offer_mat):
    '''Measure the squared errors for the prediction'''
    num_complete = np.count_nonzero(~np.isnan(test_matrix))
    
    sse_accum = 0
    
    for user_id in test_matrix.index:
        for offer_id in test_matrix.columns:
            if ~np.isnan(test_matrix.loc[user_id, offer_id]):
                predict_value = predict_reaction(user_mat, offer_mat, user_id, offer_id)
                if predict_value != None:
                    # compute the error as the actual minus the dot product of the user and offer latent features
                    diff = test_matrix.loc[user_id, offer_id] - predict_value #predict_reaction(user_mat, offer_mat, user_id, offer_id)

                    # Keep track of the sum of squared errors for the matrix
                    sse_accum += diff**2
    
    print('Validation score: ',sse_accum / num_complete)


def offer_max_reactions(user_item_matrix):
    # Find out which offer is accepted the most number of times

    offer_count = []
    for offer_id in user_item_matrix.columns:
        offer_count.append([offer_id, len(transcript_df[(transcript_df['person'].isin(list(user_item_matrix[user_item_matrix[offer_id]>=1].index)))])])

    offer_reactions = pd.DataFrame(offer_count, columns=['offer_id', 'Total_reactions'])
    offer_reactions['Total_reactions'] = pd.to_numeric(offer_reactions['Total_reactions'])
    offer_reactions.sort_values(by='Total_reactions', ascending=False, inplace=True)
    
    return offer_reactions

def make_recommendations(user_id, user_mat, offer_mat):
    reccomendations = {}
    for offer_id in train_matrix.columns:
        pred_val = predict_reaction(user_mat, offer_mat, user_id, offer_id)
        if pred_val != None:
            reccomendations[offer_id] = pred_val
    if pred_val == None:
        print("Since this user is new, we are reccomending the best performing offer")
        print(offer_reactions.head(1))
    else:
        import operator
        from more_itertools import take
        reccomendations= dict( sorted(reccomendations.items(), key=operator.itemgetter(1),reverse=True))
        top_reccomendation = take(1, reccomendations.items())
        for offer_id, pred_val in top_reccomendation:
            print("offer id: ", offer_id, " predicted value: ", round(pred_val,2))