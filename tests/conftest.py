"""Fixture policy is always explicit in CI; local default remains exact."""
import pytest


def pytest_addoption(parser):
    parser.addoption('--fixture-mode', choices=('exact', 'portable'), default='exact',
                     help='Fixture regression policy (default: exact)')


def pytest_report_header(config):
    return 'Fixture comparison mode: ' + config.getoption('--fixture-mode')


@pytest.fixture
def fixture_mode(request):
    mode = request.config.getoption('--fixture-mode')
    request.node.user_properties.append(('fixture_mode', mode))
    return mode
