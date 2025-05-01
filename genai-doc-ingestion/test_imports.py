#!/usr/bin/env python
"""
Test script to validate our import fixes.
"""

def test_servicenow_connector():
    print("Testing ServiceNowConnector...")
    try:
        from app.services.connectors.servicenow_connector import ServiceNowConnector
        connector = ServiceNowConnector()
        print("  ✓ ServiceNowConnector imported and initialized successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_sharepoint_connector():
    print("Testing SharePointConnector...")
    try:
        from app.services.connectors.sharepoint_connector import SharePointConnector
        connector = SharePointConnector()
        print("  ✓ SharePointConnector imported and initialized successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_hallucination_detector():
    print("Testing HallucinationDetector...")
    try:
        from app.services.quality.hallucination_detector import HallucinationDetector
        detector = HallucinationDetector()
        print("  ✓ HallucinationDetector imported and initialized successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_ab_testing():
    print("Testing ABTestingManager...")
    try:
        from app.services.experiment.ab_testing import ABTestingManager, AllocationStrategy
        manager = ABTestingManager()
        print("  ✓ ABTestingManager imported and initialized successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_api_routes():
    print("Testing API routes...")
    try:
        from app.routes.api import router
        print("  ✓ API routes imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Running import tests...")
    
    servicenow_success = test_servicenow_connector()
    sharepoint_success = test_sharepoint_connector()
    hallucination_success = test_hallucination_detector()
    ab_testing_success = test_ab_testing()
    api_routes_success = test_api_routes()
    
    print("\nSummary:")
    print(f"ServiceNowConnector: {'✓' if servicenow_success else '✗'}")
    print(f"SharePointConnector: {'✓' if sharepoint_success else '✗'}")
    print(f"HallucinationDetector: {'✓' if hallucination_success else '✗'}")
    print(f"ABTestingManager: {'✓' if ab_testing_success else '✗'}")
    print(f"API Routes: {'✓' if api_routes_success else '✗'}")
    
    total_success = servicenow_success and sharepoint_success and hallucination_success and ab_testing_success and api_routes_success
    
    print(f"\nOverall: {'✓ All tests passed!' if total_success else '✗ Some tests failed!'}") 