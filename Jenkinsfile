#!groovy
def BN = (BRANCH_NAME == 'master' || BRANCH_NAME.startsWith('releases/')) ? BRANCH_NAME : 'releases/2025-07'

def repositoryName = 'knime-timeseries-analysis'

library "knime-pipeline@$BN"

properties([
    parameters(knimetools.getPythonExtensionParameters()),
    buildDiscarder(logRotator(numToKeepStr: '5')),
    disableConcurrentBuilds()
])

try {
    knimetools.defaultPythonExtensionBuild()

    workflowTests.runTests(
        dependencies: [
            repositories: [
                'knime-python',
                'knime-python-types',
                'knime-core-columnar',
                'knime-testing-internal',
                'knime-python-legacy',
                'knime-conda',
                'knime-conda-channels',
                'knime-credentials-base',
                'knime-gateway',
                'knime-base',
                'knime-productivity-oss',
                'knime-json',
                'knime-javasnippet',
                'knime-reporting',
                'knime-filehandling',
                'knime-scripting-editor',
                repositoryName
                ],
            ius: [
                'org.knime.features.core.columnar.feature.group'
            ]
        ]
    )
} catch (ex) {
    currentBuild.result = 'FAILURE'
    throw ex
} finally {
    notifications.notifyBuild(currentBuild.result)
}
