# IRenE Webapp 🚀

**IRenE (Iterative Refinement and Exploration)** is a powerful web tool designed to help researchers discover relevant scholarly articles based on an initial seed corpus of documents. Leveraging the Scopus API and natural language processing techniques, IRenE performs iterative sampling to provide a ranked list of articles that closely align with the user's research interests.

## 🌟 Features

- **Seed Corpus Upload**: Upload one or multiple PDF files to kickstart your research exploration.
- **Keyword Extraction**: Automatically extract significant keywords from uploaded documents to guide and optimize the search.
- **Scopus API Integration**: Utilize your own Scopus API key for authenticated, personalized searches. [Generate your key here](https://dev.elsevier.com/).
- **Iterative Sampling**: Configure iterations and thresholds to fine-tune the sampling process for optimal results.
- **Real-Time Progress Updates**: Visual progress bar and detailed status updates, showing current queries and match counts during the sampling process.
- **Detailed Results Display**: View a ranked, scrollable table of relevant papers, with additional information such as:
  - 📊 **Occurrences**: How often the paper was retrieved during sampling.
  - 👤 **First Author**
  - 🗓️ **Year**
  - 📝 **Title**
  - 📚 **Journal/Publication**
  - 🔢 **Citations**
  - 🔓 **Open Access Status**
  - 🔗 **Direct Link to Scopus**
- **📥 CSV Export**: Download the ranked results as a CSV file for offline analysis, with proper encoding to handle special characters.

## 💡 Concept

IRenE Webapp analyzes a seed corpus to extract keywords that represent the user's core topics of interest. These keywords are used to perform iterative searches on the Scopus database, refining the results through configurable parameters such as the number of iterations and thresholds. The goal is to expand the search space while maintaining relevance, ultimately delivering a comprehensive list of pertinent scholarly articles.

## 🛠️ Roadmap

The following are key priorities and upcoming enhancements for IRenE Webapp:

- **⚙️ Error Handling**: Implement robust error-handling mechanisms for better stability and user experience, ensuring graceful recovery from API errors and invalid inputs.
- **🔍 Testing**: Develop comprehensive unit and integration tests to ensure reliability, maintainability, and prevent regressions.
- **📚 Documentation**: Provide detailed user guides and developer documentation to assist both users and contributors.
- **☁️ Deployment to Heroku**: Prepare and deploy the application on Heroku, allowing users to access IRenE directly from their browsers without the need for local installation.

## 📜 Citation

If you use IRenE Webapp in your research, please cite the following publication:

> Lechtenberg, F., Farreres, J., Galvan-Cara, A.-L., Somoza-Tornos, A., Espuña, A., & Graells, M. (2022). Information retrieval from scientific abstract and citation databases: A query-by-documents approach based on Monte-Carlo sampling. *Expert Systems with Applications, 199*, 116967. https://doi.org/10.1016/j.eswa.2022.116967

---

IRenE Webapp empowers researchers by simplifying the process of finding relevant scholarly works, all while offering flexible configurations, real-time feedback, and a seamless integration with Scopus. 🎓🔬
