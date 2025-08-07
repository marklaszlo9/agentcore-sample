import { UserManager } from "oidc-client-ts";

const cognitoAuthConfig = {
    authority: "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_O6RLwBTHN",
    client_id: "734547mm50iohgjf1is1oa36qc",
    redirect_uri: "https://d1a1j08hw40vr1.cloudfront.net",
    response_type: "code",
    scope: "phone openid email"
};

// Create a UserManager instance
export const userManager = new UserManager({
    ...cognitoAuthConfig,
});

export async function signOutRedirect() {
    try {
        // For Cognito without a configured hosted UI domain, we'll just do local signout
        // This clears the tokens and redirects to the main page
        await userManager.removeUser();
        
        // Clear any additional session storage
        sessionStorage.clear();
        localStorage.removeItem('oidc.user:' + cognitoAuthConfig.authority + ':' + cognitoAuthConfig.client_id);
        
        // Redirect to home page
        window.location.href = cognitoAuthConfig.redirect_uri;
    } catch (error) {
        console.error('Error during signout:', error);
        // Force redirect anyway
        window.location.href = cognitoAuthConfig.redirect_uri;
    }
}

export async function getCurrentUser() {
    try {
        return await userManager.getUser();
    } catch (error) {
        console.error('Error getting current user:', error);
        return null;
    }
}

export async function signIn() {
    try {
        await userManager.signinRedirect();
    } catch (error) {
        console.error('Error signing in:', error);
        throw error;
    }
}

export async function handleSigninCallback() {
    try {
        const user = await userManager.signinRedirectCallback();
        return user;
    } catch (error) {
        console.error('Error handling signin callback:', error);
        throw error;
    }
}

export async function getAccessToken() {
    try {
        const user = await getCurrentUser();
        return user?.access_token;
    } catch (error) {
        console.error('Error getting access token:', error);
        return null;
    }
}

export async function signOutLocal() {
    try {
        // Just remove the user from local storage and reload the page
        await userManager.removeUser();
        window.location.reload();
    } catch (error) {
        console.error('Error during local signout:', error);
        // Force reload anyway
        window.location.reload();
    }
}