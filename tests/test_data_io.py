import unittest
import tempfile
from pathlib import Path
from experiments.data_io import select_news_rows, read_text_pack

class DataIntegrityTests(unittest.TestCase):
    def test_news_sampling_ignores_labels_and_excludes_normalized_train_overlap(self):
        train=[{'text':' A B ','label':1},{'text':'a b','label':0},{'text':'another','label':2}]
        test=[{'text':'Ａ B','label':3},{'text':'test text','label':1}]
        a,b=select_news_rows(train,test,train_n=2,test_n=1)
        c,d=select_news_rows([{**r,'label':99} for r in train],test,train_n=2,test_n=1)
        self.assertEqual([r['text'] for r in a],[r['text'] for r in c]);self.assertEqual(b,d)
        self.assertEqual(len(a),2);self.assertEqual(b[0]['text'],'test text')


    def test_model_input_rejects_label_columns_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.csv'
            for content in ['doc_id,text,label\n1,document,0\n','doc_id,text\n1,hello\n1,world\n']:
                p.write_text(content)
                with self.assertRaises(ValueError): read_text_pack(p)
            p.write_text('doc_id,text\n1,NA\n2,001\n')
            ids,text=read_text_pack(p);self.assertEqual(text,['NA','001'])


