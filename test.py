from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")
## concatenate all data to guarantee that dataset have the same columns
all_data = pd.concat([train, holdout], axis=0)

class DataFiller(BaseEstimator, TransformerMixin):
    """
    Applies data filling to NaN values in selected features.
    
    > cols_filler: dictionary with columns and filling values / strategy
    e.g. {"A": 0.5, "B": -2, "C": 'a', "D": 'mean'}
    """
    def __init__(self, cols_filler):
        """
        Initial input for the transformer.
        """
        self.cols_filler = cols_filler
        pass
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        """
        Where the filling occurs.
        """
        for k, v in self.cols_filler.items():
            # Filling strategy
            if v == 'mean':
                filler = X[k].mean()
            elif v == 'median':
                filler = X[k].median()
            else:
                filler = v
            
            X[k] = X[k].fillna(filler)
            
        return X
            

class DataBinning(BaseEstimator, TransformerMixin):
    """
    Applies binnig to selected features.
    
    > dict_of_cols: dictionary of dictionaries with cut_points and labels for each column.
    e.g. {"A":{'cut_points':[1,2,3], 'labels':['a', 'b']}, "B": {...}, ...}
    """
    def __init__(self, dict_of_cols, inplace=True):
        """
        Inital input for the transformer.
        """
        self.dict_cols = dict_of_cols
        self.inplace = inplace
        pass
    
    def fit(self):
        return self
    
    def transform(self, X):
        """
        Where the binning occurs.
        """
        for k, v in self.dict_cols.items():
            
            # Cut points data
            cut_points = v['cut_points']
            
            # Labels data
            label_names = v['labels']
            
            # Creates new columns inplace
            if self.inplace:
                X[k] = pd.cut(X[k], cut_points, labels=label_names)
            else:
                k = k + '_binned'
                X[k] = pd.cut(X[k], cut_points, labels=label_names)
            
        return X
    
class DataProcess(BaseEstimator, TransformerMixin):
    """
    Applies application-specific process to selected features.
    
    """
    def __init__(self):
        pass
    
    def fit(self):
        return self
    
    def transform(self, X):
        """
        Where the processing occurs.
        """
        # Process Tickets column
        ticket_cod = []
        ticket_number = []
        for index, ticket in X.Ticket.iteritems():
            if not ticket.isdigit():
                # Take prefix
                split = ticket.replace(".","").replace("/","").strip().split(' ')
                ticket_cod.append(split[0])
                # Take ticket number
                try:
                    ticket_number.append(int(split[1]))
                except:
                    ticket_number.append(-1)
            else:
                ticket_cod.append("X")
                try:
                    ticket_number.append(int(ticket))
                except:
                    ticket_number.append(-1)
        X["Ticket_cod"] = ticket_cod
        X["Ticket_number"] = ticket_number
        X = X.drop('Ticket',axis=1)
        
        # Process titles
        titles = {"Mr" :         "Mr",
                  "Mme":         "Mrs",
                  "Ms":          "Mrs",
                  "Mrs" :        "Mrs",
                  "Master" :     "Master",
                  "Mlle":        "Miss",
                  "Miss" :       "Miss",
                  "Capt":        "Officer",
                  "Col":         "Officer",
                  "Major":       "Officer",
                  "Dr":          "Officer",
                  "Rev":         "Officer",
                  "Jonkheer":    "Royalty",
                  "Don":         "Royalty",
                  "Sir" :        "Royalty",
                  "Countess":    "Royalty",
                  "Dona":        "Royalty",
                  "Lady" :       "Royalty"}
        extracted_titles = X["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
        X["Title"] = extracted_titles.map(titles)
        X = X.drop('Name', axis=1)
        
        # Process Cabin
        cabin_cod = []
        cabin_number = []
        for index, cabin in X.Cabin.iteritems():
            if isinstance(cabin, str):
                # Take prefix
                split = cabin.strip().split(' ')[-1]
                cabin_cod.append(split[0])
                # Cabin number
                try:
                    cabin_number.append(int(split[1:]))
                except:
                    cabin_number.append(-1)
            else:
                cabin_cod.append('Unknown') 
                cabin_number.append(-1)
        X["Cabin_type"] = cabin_cod
        X["Cabin_number"] = cabin_number
        X = X.drop('Cabin',axis=1)
        
        # Is alone
        X["Family_size"] = X[["SibSp","Parch"]].sum(axis=1)
        X["Alone"] = (X["Family_size"] == 0)
        
        return X

# input dictionaries
dict_fill = { "Fare": "mean",
              "Embarked": "S",
              "Age": -0.5,
              "Fare": 'mean',
            }
dict_binning = {"Age": {"cut_points": [-5, 0, 5, 12, 18, 35, 60, 100],
                         "labels": ["Missing", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]},
                "Fare": {"cut_points": [-1, 12, 50, 100, 1000],
                         "labels": ["0-12","12-50","50-100","100+"]},
                "Cabin_number": {"cut_points": [-1, 0, 33, 67, 100, 133, 1000],
                                 "labels": ["Unknown", "0-33", "33-67", "67-100", "100-133", "133+"]},
                "Ticket_number": {"cut_points": [-1, 0, 3000, 10000, 20000, 50000, 100000, 300000, 10000000],
                                  "labels": ["Unknown", "0-3k", "3k-10k", "10k-20k", "20k-50k", "50k-100k", "100k-300k", "300k+"]}
               }
# Pipeline definition
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('filling', DataFiller(dict_fill)),
                     ('processing', DataProcess()),
                     ('binnig', DataBinning(dict_binning))
])

transformed_data = pipeline.transform(all_data)