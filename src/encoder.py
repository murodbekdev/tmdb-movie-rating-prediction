import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Encoder:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()


    # Train encoding
   

    def encodla_train(self, onehot_threshold=5):

        label_cols = {}
        onehot_cols = {}

        try:
            # categorical columns 
           

            categorical_cols = self.df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()

            logging.info(
                f"🔍 Categorical ustunlar: {categorical_cols}"
            )


         
            # label encoding va onehot encoding

            for col in categorical_cols:

                unique_count = self.df[col].nunique(
                    dropna=True
                )

                # one hot

                if unique_count <= onehot_threshold:

                    onehot_cols[col] = unique_count

                    logging.info(
                        f"🔹 One-Hot Encoding: "
                        f"{col} ({unique_count} categories)"
                    )

                # label  encoding
              
                else:

                    le = LabelEncoder()

                    # NaN bo'lsa string sifatida ishlashiga
                    # tayyorlaymiz
                    self.df[col] = (
                        self.df[col]
                        .fillna("__MISSING__")
                        .astype(str)
                    )

                    self.df[col] = le.fit_transform(
                        self.df[col]
                    )

                    label_cols[col] = le

                    logging.info(
                        f"🔹 Label Encoding: "
                        f"{col} ({unique_count} categories)"
                    )
       
            # one-hot encoding 

            if onehot_cols:

                self.df = pd.get_dummies(
                    self.df,
                    columns=list(onehot_cols.keys()),
                    drop_first=True,
                    dtype=int
                )

            # so'ngi tekshiruv
           
            remaining_categorical = self.df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()


            if remaining_categorical:

                logging.error(
                    f"❌ Encodingdan keyin categorical "
                    f"ustunlar qolib ketdi: "
                    f"{remaining_categorical}"
                )

                raise ValueError(
                    "Encodingdan keyin categorical ustunlar "
                    "qolib ketdi: "
                    f"{remaining_categorical}"
                )


            logging.info(
                "✅ Train uchun Encoding muvaffaqiyatli amalga oshirildi."
            )

            logging.info(
                f"📊 Train encoded shape: {self.df.shape}"
            )

            return (
                self.df,
                label_cols,
                onehot_cols
            )


        except Exception as e:

            logging.error(
                "❌ Train uchun Encoding "
                "muvaffaqiyatsizlikka uchradi.",
                exc_info=True
            )

            raise e


  
    # test encoding 

    def encodla_test(
        self,
        label_cols,
        onehot_cols,
        train_columns
    ):

        try:
           
            # label encoding
           

            for col, le in label_cols.items():

                self.df[col] = (
                    self.df[col]
                    .fillna("__MISSING__")
                    .astype(str)
                )

                self.df[col] = self.df[col].apply(
                    lambda x:
                    le.transform([x])[0]
                    if x in le.classes_
                    else -1
                )
           
            # one-hot encoding
           

            if onehot_cols:

                self.df = pd.get_dummies(
                    self.df,
                    columns=list(onehot_cols.keys()),
                    drop_first=True,
                    dtype=int
                )
           
            # Train columns bilan bir hil qilish 
           
            self.df = self.df.reindex(
                columns=train_columns,
                fill_value=0
            )
           
            # so'ngi tekshiruv
           
            remaining_categorical = self.df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()


            if remaining_categorical:

                logging.error(
                    f"❌ Test encodingdan keyin categorical "
                    f"ustunlar qolib ketdi: "
                    f"{remaining_categorical}"
                )

                raise ValueError(
                    "Test encodingdan keyin categorical "
                    f"ustunlar qolib ketdi: "
                    f"{remaining_categorical}"
                )


            logging.info(
                "✅ Test uchun Encoding muvaffaqiyatli amalga oshirildi."
            )

            logging.info(
                f"📊 Test encoded shape: {self.df.shape}"
            )

            return self.df


        except Exception as e:

            logging.error(
                "❌ Test uchun Encoding "
                "muvaffaqiyatsizlikka uchradi.",
                exc_info=True
            )

            raise e