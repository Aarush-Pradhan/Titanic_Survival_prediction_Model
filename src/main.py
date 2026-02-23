from preprocessing import load_data, clean_data
from Train import train_model
from evaluate import evaluate_model
from visualization import plot_confusion_matrix, plot_survival_by_gender

def main():

    print("Titanic Survival Prediction Project Starting...\n")

    df=load_data("../data/titanic.csv")
    print("data loaded succesfully....")

    df = clean_data(df)
    print(" Data cleaning completed....")

    model,x_test,y_test = train_model(df)
    print("Model training completed....")

    evaluate_model(model,x_test,y_test)

    plot_confusion_matrix(model, x_test, y_test)
    plot_survival_by_gender(df)



    print("\n🎯 Project Completed Successfully!")


if __name__ == "__main__":
    main()
