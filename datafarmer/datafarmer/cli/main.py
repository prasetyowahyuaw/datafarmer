import typer
from datafarmer.io import read_bigquery, write_bigquery
from datafarmer.utils import logger
from datafarmer.llm import Gemini
from rich import print
from rich.prompt import Prompt

app = typer.Typer()

import warnings
warnings.filterwarnings("ignore")

@app.command()
def test():
    """test the cli command"""
    print(f"Yay! the :ear_of_rice: [bold green]Datafarmer Gemini[/bold green] :ear_of_rice: cli command is working :rocket:")

@app.command()
def gemini():
    """
    this command will generate llm output from bigquery data
    """

    print(":ear_of_rice: [bold green]Datafarmer Gemini[/bold green] :ear_of_rice: \n[cyan1]Generating llm output from bigquery data. Please enter the required below[/cyan1]")
    project_id = Prompt.ask("[cyan1]- Enter the google project id :pray:[/cyan1]")
    table_name = Prompt.ask("[cyan1]- Enter the table name with the dataset :pray:[/cyan1]")
    destination_name = Prompt.ask("[cyan1]- Enter the destination dataset and table name for the output :pray: (blank for default)[/cyan1]")
    print("[cyan1]Thanks, let me help you :rocket:[/cyan1]")

    logger.info(f"Load data from bigquery `{project_id}`.`{table_name}`")
    try:
        
        # read data from bigquery
        data = read_bigquery(query=f"SELECT * FROM `{project_id}`.`{table_name}`", project_id=project_id)
        logger.info(f"Data loaded from bigquery")

        # generation process
        gemini = Gemini(project_id=project_id)
        output = gemini.generate_from_dataframe(data) # dataframe with id and result columns

        # write the output to bigquery
        destination_name = destination_name if destination_name else f"{table_name}_output"
    
        write_bigquery(
            df=output,
            project_id=project_id,
            table_id=destination_name.split(".")[1],
            dataset_id=destination_name.split(".")[0],
            mode="WRITE_TRUNCATE",
        )
        logger.info(f"Output saved to bigquery `{project_id}`.`{destination_name}`")

        print("[green]Awesome! the generation already done :tada:[/green]")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        print("[red]Oops! something went wrong :sob:[/red]")

if __name__ == "__main__":
    app()