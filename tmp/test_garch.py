from data.loader import load_asset
from models.garch import forecast


def main():
    df = load_asset('RELIANCE.NS', '1y', '1d')
    res = forecast(df, 7)
    print(res['regime'])
    print(res['metrics'])
    print(res['forecast_df'].head().to_string(index=False))


if __name__ == '__main__':
    main()
