from libc.stdlib cimport rand, RAND_MAX
import numpy as np
cimport numpy as np
    

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE_INT = np.int
DTYPE_FLOAT = np.float

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_INT_T
ctypedef np.float_t DTYPE_FLOAT_T



def get_argmax(np.ndarray[DTYPE_FLOAT_T, ndim=2] target_matrix):
    
    cdef int best_row, best_col
    cdef int row_index, col_index
    cdef DTYPE_FLOAT_T best_score = -np.inf
    
    cdef int ROW = target_matrix.shape[0]
    cdef int COL = target_matrix.shape[1]
    
    for row_index in range(ROW):
        for col_index in range(COL):
            
            if target_matrix[row_index, col_index] > best_score:
                best_score = target_matrix[row_index, col_index]
                best_row = row_index
                best_col = col_index
                
    return (best_row, best_col, best_score)



def get_hamming_loss(np.ndarray[DTYPE_INT_T] z1, np.ndarray[DTYPE_INT_T] z2, int relation_num):
    
    cdef int rel_index
    cdef DTYPE_FLOAT_T loss = 0.0
    
    for rel_index in range(relation_num):
        
        if z1[rel_index] != z2[rel_index]:
            loss += 1.0
    
    return loss




def get_max_index(np.ndarray[DTYPE_FLOAT_T, ndim=1] target_matrix):
    
    cdef DTYPE_FLOAT_T best_score = -np.inf
    cdef DTYPE_INT_T best_index = -1
    
    cdef int ROW = target_matrix.shape[0]
    
    for row_index in range(ROW):
            
        if target_matrix[row_index] > best_score:
            best_score = target_matrix[row_index]
            best_index = row_index
         
    return best_score, best_index




def get_secondeMax_index(np.ndarray[DTYPE_FLOAT_T, ndim=1] target_matrix):
    
    cdef DTYPE_FLOAT_T best_score = -np.inf
    cdef DTYPE_INT_T best_index = -1
    
    cdef DTYPE_FLOAT_T secondeMax_score = -np.inf
    cdef DTYPE_INT_T secondeMax_index = -1
    
    cdef int ROW = target_matrix.shape[0]
    
    if ROW < 2:
        print "No second max since only one element"
    
    for row_index in range(ROW):
        
        if target_matrix[row_index] > best_score:
            
            secondeMax_score = best_score
            secondeMax_index = best_index
            
            best_score = target_matrix[row_index]
            best_index = row_index
         
    return secondeMax_score, secondeMax_index




def loss_augmented_search(np.ndarray[DTYPE_FLOAT_T, ndim=2] post_z, np.ndarray[DTYPE_INT_T] z_star, int relation_num):

    cdef int sen_num = post_z.shape[0]

    cdef np.ndarray[DTYPE_INT_T] best_z = np.zeros(sen_num, dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_INT_T] best_rel = np.zeros(relation_num, dtype=DTYPE_INT)
    
    
    cdef int index_z, max_index, secondeMax_index
    cdef DTYPE_FLOAT_T max_score, secondeMax_score
    
    
    for index_z in range(sen_num):
        
        max_score, max_index = get_max_index(post_z[index_z])
        secondeMax_score, secondeMax_index = get_secondeMax_index(post_z[index_z])
        
        # print max_score, max_index
        # print secondeMax_score, secondeMax_index
        
        if max_index != z_star[index_z] or max_score - secondeMax_score >= 1:
            best_z[index_z] = max_index
        else:
            best_z[index_z] = secondeMax_index
            
        best_rel[best_z[index_z]] = 1
        
        
    return best_z, best_rel
    


def local_search(np.ndarray[DTYPE_FLOAT_T, ndim=2] post_z, np.ndarray[DTYPE_FLOAT_T] rel_penality, int num_rand_restart, int relation_num):

    cdef int sen_num = post_z.shape[0]

    cdef np.ndarray[DTYPE_INT_T] best_z = np.zeros(sen_num, dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_INT_T] best_rel = np.zeros(relation_num, dtype=DTYPE_INT)
    cdef DTYPE_FLOAT_T best_score = -np.inf
    cdef DTYPE_FLOAT_T score
    
    cdef np.ndarray[DTYPE_INT_T] init_z
    cdef np.ndarray[DTYPE_INT_T] rel, rel_count
    cdef np.ndarray[DTYPE_FLOAT_T, ndim=2] deltas
    cdef np.ndarray[DTYPE_FLOAT_T, ndim=2] deltas_aggregate
    
    cdef DTYPE_FLOAT_T delta, delta_aggregate
    cdef DTYPE_FLOAT_T hamming_loss
    
    cdef int i, index_z, z_value, index_r, r_value
    cdef int rel_index, sen_index, sen_index_2, sen_index_3, r1, r2
    cdef int new_rel_index, new_rel
    cdef int rel1, rel2
    

    for i in range(num_rand_restart):

        rel = np.zeros(relation_num, dtype=DTYPE_INT)
        rel_count = np.zeros(relation_num, dtype=DTYPE_INT)
        score = 0.0


        # Random initialization
        # np.random.seed(i*10)
        init_z = np.random.randint(relation_num, size=sen_num, dtype=DTYPE_INT)
        # if sen_num == 4:
        #     print init_z
            
        for index_z in range(sen_num):
            z_value = init_z[index_z]
            # Add up the score 
            score += post_z[index_z, z_value] 
            # Increase the relation count
            rel_count[z_value] += 1
            # Lable the specific relation
            rel[z_value] = 1
            
            
        for index_r in range(relation_num):
            if rel_count[index_r] > 0:
                # Add up the corresponding award for the relation choice
                score += rel_penality[index_r]
                

        stayed_same = False

        while not stayed_same:

            # First search operator (change one variable)
            deltas = np.zeros((sen_num, relation_num), dtype=DTYPE_FLOAT)
            # print deltas.shape
            
            
            for rel_index in range(relation_num):
                for sen_index in range(sen_num):

                    # If not the random choice
                    if rel_index != init_z[sen_index]:

                        # Caculate the cost of moving to this relation
                        deltas[sen_index, rel_index] = post_z[sen_index, rel_index] - post_z[sen_index, init_z[sen_index]]  

                        # If the relation not exists
                        if rel_count[rel_index] == 0:
                            deltas[sen_index, rel_index] += rel_penality[rel_index]

                        # If r onlys exists once
                        if rel_count[init_z[sen_index]] == 1:
                            deltas[sen_index, rel_index] -= rel_penality[init_z[sen_index]]


            # Second search operator (switch all instances of relation r to NA)
            deltas_aggregate = np.zeros((relation_num, relation_num), dtype=DTYPE_FLOAT)

            for r1 in range(relation_num):
                for r2 in range(relation_num):

                    # Two different relations r1 & r2, r1 already exists
                    if rel_count[r1] > 0 and r1 != r2:
                        for sen_index_2 in range(sen_num):
                            if init_z[sen_index_2] == r1:
                                # Aggregate loss of changing from r1 to r2
                                deltas_aggregate[r1, r2] += post_z[sen_index_2, r2] - post_z[sen_index_2, r1]

                        # Remove the award from r1
                        deltas_aggregate[r1, r2] -= rel_penality[r1] 
                        # If r2 not exist, add up the award from r2
                        if rel_count[r2] == 0:
                            deltas_aggregate[r1, r2] += rel_penality[r2] 
                            
                            
            stayed_same = True
            
            # Numpy function is kind of slow
            # (new_rel_index, new_rel) = np.unravel_index(deltas.argmax(), (sen_num, relation_num))
            (new_rel_index, new_rel, delta) = get_argmax(deltas)
            
            old_rel = init_z[new_rel_index]
            # delta = deltas[new_rel_index, new_rel]
            
            # delta_aggregate = deltas_aggregate.max()
            (rel1, rel2, delta_aggregate) = get_argmax(deltas_aggregate)

            # Check which search operator provides the greatest score delta
            if delta_aggregate > delta and delta_aggregate > 0:

                # Change all instances of the max deltaNA relation to NA
                score += delta_aggregate

#                 (r1, r2) = np.unravel_index(deltas_aggregate.argmax(), deltas_aggregate.shape)

                for sen_index_3 in range(sen_num):
                    if init_z[sen_index_3] == rel1:
                        init_z[sen_index_3] = rel2
                        rel_count[rel2] += 1

                rel_count[rel1] = 0
                rel[rel1] = 0
                rel[rel2] = 1
                stayed_same = False

            elif old_rel != new_rel and delta > 0:
                score += delta
                init_z[new_rel_index] = new_rel
                rel_count[new_rel] += 1
                rel[new_rel] = 1
                rel_count[old_rel] -= 1
                if rel_count[old_rel] == 0:
                    rel[old_rel] = 0
                    
                stayed_same = False        


        if score > best_score:
            best_score = score
            best_z = init_z
            best_rel = rel


    return best_z, best_rel