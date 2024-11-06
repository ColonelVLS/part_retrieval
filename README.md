# part_retrieval

Το πρότζεκτ αποτελείται από 3 βασικά αρχεία: 
 
* CLIP_chamfer.ipynb Εδώ γίνεται η εκπαίδευση του μοντέλου
* Pre_encoding.ipynb Εδώ το μοντέλο κάνει ένα πέρασμα για να κάνει encode όλα τα samples του dataset, ώστε το feature vector να είναι διαθέσιμο στο runtime
* Retrieval.ipynb Εδώ είναι το use case όπου δίνουμε ως είσοδο ένα σχήμα (χωρισμένο σε parts) και το μοντέλο κάνει ένα πέρασμα τη βάση για να sortάρει όλα τα samples ανάλογα με την καταλληλότητα τους.
