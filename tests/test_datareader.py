from LSTM_for_Stock.data_processor import get_ipo_date
import LSTM_for_Stock.data_processor as data_processor
import datetime


def test_get_ipo_date():
    print(data_processor._get_tushare_ipo_date('000002'))
    print(data_processor._get_tdx_ipo_date('000002'))
    d = get_ipo_date('000002')
    assert d is not None
    assert isinstance(d, datetime.datetime)
    print(get_ipo_date('399300'))


def test_get_block_code():
    lst = data_processor.get_block_code('000002')
    assert lst is not None
    assert '000002' not in lst
