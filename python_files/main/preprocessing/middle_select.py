####Create python function that selects a certain range of data to subset the training data. The output is the final list of values to take. Useful for selecting the middle N-2n of each sample.

#requires training_data to be loaded into original file.

#N = size of basepairs = 29846
#n = size of cutoff at ends. Can try anywhere from 0 (full set) to 14922 (middle 2 bases).
#M = sets used for the training data.
def subset_list(n,N,M):
    sublist = []
    for m in range(0,M):
        stand_in = [*range(N*m+(n),(N*(m+1)-1)-(n-1))]
        sublist = sublist+stand_in
    
    return sublist


