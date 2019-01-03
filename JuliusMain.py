import IOHandler as io
import PreprocessingHandler as ph

data = io.read_data('train')
ph.hot_encode_columns(data,'train')
