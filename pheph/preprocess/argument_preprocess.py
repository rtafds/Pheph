import numpy as np
import pandas as pd
from pandas.api import types
import re

class ArgumentOrganizer():

    # 個数によるカテゴリ or 連続値判定
    @staticmethod
    def _define_threshold(
        x,
        a=9.09762422e01,
        b=3.06669949e05,
        c=1.47003672e-18,
        d=1.34823829e-08,
        e=-1.41826938e01,
    ):
        # x = np.array([10,10e2,10e3,10e4,10e5,10e6,10e7,10e8])
        # y = np.array([6,30,50,70,90,100,120,150])
        # を目安にカーブフィッティング
        return a * np.log(x) / np.log(b + 1) + c * x * x + d * x + e

    @staticmethod
    def _define_threshold2(
        x,
        a=4.01935067e02,
        b=1.05881395e02,
        c=9.91024423e01,
        d=1.67532366e-06,
        e=-8.64059054e01,
    ):
        # 　x = np.array([10,10e2,10e3,10e4,10e5,10e6,10e7,10e8])
        # 　y = np.array([5,40,80,120,150,200,350,500])
        # を目安にカーブフィッティング
        return a * 1 / x + b * np.log(x) / np.log(c + 1) + d * x + e

    @classmethod
    def is_continuous_df(cls, data, threshold="auto"):
        """連続値かどうかを判定する。threshold以上のものを連続値と判定する。
        Args:
            data (pandas.DataFrame): 使用するデータ
            threshold (int, optional): 閾値。threshold以上のデータを連続値として判定. Defaults to "auto".
            "auto", "auto2" の場合、事前に用意した関数によりいい感じに判定される。"auto2"の方がカテゴリに判定する範囲が広い。
            "auto"の方がカテゴリに判定する範囲が狭い。厳し目。

        Returns:
            list[bool] : 連続値かどうかの判定リスト。
        """
        row_num = data.shape[0]
        unique_counts = data.nunique()

        if threshold == "auto":
            threshold = cls._define_threshold(row_num)
        elif threshold == "auto2":
            threshold = cls._define_threshold2(row_num)

        is_continuous_list = []
        is_continuous_list_append = is_continuous_list.append  # わずかな高速化
        for unique_count in unique_counts:
            if unique_count <= threshold:
                is_continuous_list_append(False)
            else:
                is_continuous_list_append(True)

        return is_continuous_list

    @staticmethod
    def _check_numerical_in_string(string):
        if isinstance(string,str):
            pattern = re.compile(r'\d')

            match = pattern.search(string)
            if match:
                is_in_numerical = True
            else:
                is_in_numerical = False
        else:
            is_in_numerical = False
        return is_in_numerical

    @classmethod
    def _check_is_string_in_numerical_df(cls, data):
        return data.applymap(cls._check_numerical_in_string).any()

    @staticmethod
    def judge_ctype(dtype, is_coutinuous, is_string_in_numerical):
        """random系で生成する時に、使用するデータのタイプを判定する。
        ctype(convert type) : "c": category variable, "d": dummy variable, 
        "n": nochange(float continuous, int continuous, int category, complex),"t":datetime continuous (datetime catetory is "c")
        Args:
            dtype (numpy.dtype): numpyのdtype
            float_is_int (bool): float_is_integerの結果の一要素
            is_coutinuous (bool): is_continuous_dfの結果の一要素

        Returns:
            list[str]: ctypeのリスト
        """

        # dtypeの全種類
        # int8, int16, int32, int64, uint8, uint16, uint32, uint64m, float16,
        # float32, float64, float128, complex64, complex128,complex56, bool, unicode, object
        # unicodeと変なdatetime型は設定してない。
        if types.is_float_dtype(dtype):
            ctype = "n"  # floatカテゴリ
        elif types.is_integer_dtype(dtype) or types.is_unsigned_integer_dtype(dtype):
            ctype = "n"
        elif types.is_bool_dtype(dtype):
            ctype = "c"
        elif types.is_string_dtype(dtype):
            if is_coutinuous:
                ctype = "c"
            else:
                if is_string_in_numerical:
                    ctype = "c"
                else:
                    ctype = "d"        
        elif types.is_datetime64_any_dtype(dtype):
            if is_coutinuous:
                ctype = "t"
            else:
                ctype = "c"
        elif types.is_complex_dtype(dtype):
            ctype = "n"
        else:
            raise ValueError("Using dtype is not exist.")
        return ctype
    

    @classmethod
    def judge_ctype_df(cls, data, threshold="auto"):
        """ctypeをdfに対して判定する。

        Args:
            data (pandas.DataFrame): 判定するデータフレーム
            threshold (str, optional): is_continuous_df の閾値。. Defaults to "auto". ユニークな数が何個以上だと連続値だと判定するかを設定。
            デフォルトでは、要素数から適切そうな閾値を判定される。

        Returns:
            list : 各カラムのctypeのリスト
        """
        # 型を判定
        dtype_list = data.dtypes

        # データのユニーク個数によって、連続値かどうかを判定する
        is_coutinuous_list = cls.is_continuous_df(data, threshold=threshold)

        # stringの中に数値が入っているかを判定する。入っていたらカテゴリ、なければダミーをデフォルトにしたい。
        is_string_in_numerical_list =  cls._check_is_string_in_numerical_df(data)
        
        # ctypeの判定
        n_col = data.shape[1]
        ctype_list = []
        for i in range(n_col):
            dtype = dtype_list[i]
            is_coutinuous = is_coutinuous_list[i]
            is_string_in_numerical = is_string_in_numerical_list[i]
            
            ctype = cls.judge_ctype(dtype, is_coutinuous, is_string_in_numerical)
            ctype_list.append(ctype)
        return ctype_list

    @staticmethod
    def _convert_colname_list_to_colnum_list(data_columns, colname_list):
        """列名のリストを列番号に変換する

        Args:
            data_columns (pandas.DataFrame.columns): pandas.DataFrame 型のdfに対して、df.columns をしたもの。
            colnames_list (list[str]): 列名のリスト。列番号が混じっていても良い。

        Returns:
            list[int]: 列番号のリスト
        """
        if all([isinstance(x,int) for x in colname_list]) or colname_list==[]:
            return colname_list


        colnum_list = []
        colnum_list_append = colnum_list.append

        for colname in colname_list:
            if isinstance(colname, str):
                colnum = data_columns.get_loc(colname)
            elif isinstance(colname, int):
                colnum = colname
            else:
                colnum = int(colname)
            colnum_list_append(colnum)
        return colnum_list

    @staticmethod
    def _convert_colname_dictkey_to_colnum_dictkey(data_columns, colname_dict):
        """列名のキーの辞書を列番号のキーの辞書に変換する

        Args:
            data_columns (_type_): _description_
            colname_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        dict_key_list = list(colname_dict.keys())
        if all([isinstance(x,int) for x in dict_key_list]) or colname_dict=={}:
            return colname_dict

        for colname in dict_key_list:
            if isinstance(colname, str):
                colnum = data_columns.get_loc(colname)
                colname_dict[colnum] = colname_dict.pop(colname)
            elif isinstance(colname, int):
                pass
            else:
                colnum = int(colname)
                colname_dict[colnum] = colname_dict.pop(colname)
        return colname_dict

    @classmethod
    def argument_organize(cls, data, dummies, category_le_list, category_assign_dict):
        """ダミー変数、カテゴリ変数などの処理が被ったり、指定されていない場合に自動で調整する。
        Args:
            data (pandas.DataFrame): 使用するデータ。
            dummies (list[int or str]): ダミー変数化する列番号または列名
            category_le_list (list[int or str]): Label Encorder でカテゴリ変数化する列番号または列名
            category_assign_dict (dict): 指定してカテゴリ変数化する辞書。形式は以下。
                Example : category_assign_dict = {2:{"0回":0,"1-2回":1,"3-5回":2,"6回以上":3,np.nan:4}, 3:{"0円":0,"8000円未満":1,"30000円未満":2,"30000円超":3}}


        Returns:
            dummies, category_le_list, category_assign_dict : 調整された引数を返す。
        """
        # 全て列番号に変換する
        data_columns = data.columns
        dummies = cls._convert_colname_list_to_colnum_list(data_columns, dummies)
        category_le_list = cls._convert_colname_list_to_colnum_list(data_columns, category_le_list)
        category_assign_dict = cls._convert_colname_dictkey_to_colnum_dictkey(data_columns, category_assign_dict)
        
        # setにしておく
        category_le_set = set(category_le_list)
        category_assign_keys_set = set(category_assign_dict.keys())
        dummies_set = set(dummies)

        # 優先順位 category_assign_dict -> dummies -> category_le_dict
        # category_dict と category_assign_dict が被ってる場合 -> category_le_dictから値を削除
        category_assign_cap_le = list(category_assign_keys_set.intersection(category_le_set))
        [category_le_list.remove(cap) for cap in category_assign_cap_le]

        # dummies と category_assign_dict が被っている場合 -> dummiesから値を削除
        dummies_cap_category_assign = list(dummies_set.intersection(category_assign_keys_set))
        for cap in dummies_cap_category_assign:
            del category_assign_dict[cap] 

        # dummies と category_le_list が被っている場合 -> cateogory_le_listから削除
        dummies_cap_category_le = list(dummies_set.intersection(category_le_list))
        [category_le_list.remove(cap) for cap in dummies_cap_category_le]

        # その他のカラムで "n" 以外になっているもの -> "d":dummiesに追加, "c":category_le_dictに値を追加

        default_ctype = cls.judge_ctype_df(data)
        all_set = list(dummies_set | category_assign_keys_set | category_le_set)
        default_ctype_index = [x for x in range(len(default_ctype))]
        non_definiion_index = list(set(default_ctype_index) - set(all_set))

        for index in non_definiion_index:
            ctype = default_ctype[index]
            if ctype=="c":
                category_le_list += [index]
            elif ctype=="d":
                dummies += [index]
            else:
                pass
        dummies = sorted(dummies)
        category_le_list = sorted(category_le_list)
        
        return dummies, category_le_list, category_assign_dict


            
